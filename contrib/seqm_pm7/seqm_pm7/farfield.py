"""距離による打ち切り（遠い原子対を点電荷の項だけで扱う）

PM7 では feathering（core.feather）により、7 Å 以上の原子対の二中心積分・電子–芯引力・核間反発の γ は
すべて点電荷 γ_AB = ev / R_AB（bohr）になる。2 電子積分は (μμ|λλ) = γ_AB だけが残り、他はゼロ。
核間反発に残る指数関数の項（e^{-αR} など）は 7 Å で 1e-10 程度以下、共鳴積分 β·S も S が指数関数的に小さい。

そこで、打ち切り距離 R_c（≥ 7 Å）以上の原子対を PYSEQM の原子対リストから外し（pair_outer_cutoff）、
外した対の寄与を次の点電荷の項として足し戻す。
  Hcore の対角   : H_μμ(A) += −Σ_B Z_B γ_AB
  Fock の対角    : F_μμ(A) += Σ_B N_B γ_AB       （N_B = Σ_{λ∈B} P_λλ）
  Fock の非対角  : F_μλ(A,B) += −½ γ_AB P_μλ    （交換項）
  核間反発       : Σ_{A<B} Z_A Z_B γ_AB          （calculator 側で生成熱と力に加える）
省く近似は「R ≥ R_c で S = 0、核間反発の減衰項 = 0」だけ。PM6 には feathering が無いので近似になる。
"""
import contextlib, threading
import torch
import seqm.seqm_functions.scf_loop as _SL
from . import core

_state = threading.local()
_ORIG_HCORE = _SL.hcore
_ORIG_BUILD = None


def _get():
    return getattr(_state, "ctx", None)


@contextlib.contextmanager
def active(cutoff):
    old = _get(); _state.ctx = {"cutoff": float(cutoff)} if cutoff else None
    try:
        yield
    finally:
        _state.ctx = old


def gamma_matrix(coords, real, cutoff, lcf):
    """遠い対の γ（eV）。coords (nmol, n, 3) Å、real (nmol, n) bool。打ち切りの判定は PYSEQM の Parser と同じ式"""
    d = coords.unsqueeze(1) - coords.unsqueeze(2)                     # [m, a, b] = x_b − x_a
    d2 = d.square().sum(-1)
    n = coords.shape[1]
    eye = torch.eye(n, dtype=torch.bool, device=coords.device)
    far = real.unsqueeze(1) & real.unsqueeze(2) & ~eye & (d2 >= cutoff ** 2)
    r = torch.sqrt(torch.where(far, d2, torch.ones_like(d2))) * lcf
    return torch.where(far, core._EV / r, torch.zeros_like(r)), far


def _norb(molecule, nbf):
    n = molecule.molsize
    a = torch.arange(n, device=molecule.coordinates.device).unsqueeze(0)
    nsh = molecule.nSuperHeavy.unsqueeze(1); nh = molecule.nHeavy.unsqueeze(1); nhy = molecule.nHydro.unsqueeze(1)
    if nbf == 9:
        k = torch.where(a < nsh, 9, torch.where(a < nsh + nh, 4, torch.where(a < nsh + nh + nhy, 1, 0)))
    else:
        k = torch.where(a < nh, 4, torch.where(a < nh + nhy, 1, 0))
    return k                                                           # (nmol, molsize) 実際の軌道数


def hcore(molecule):
    out = _ORIG_HCORE(molecule)
    ctx = _get()
    if ctx is None:
        return out
    M = out[0]
    nmol, n = molecule.nmol, molecule.molsize
    nbf = M.shape[-1]
    real = molecule.species > 0
    gam, far = gamma_matrix(molecule.coordinates, real, ctx["cutoff"], molecule.const.length_conversion_factor)
    tore = molecule.const.tore[molecule.species]                        # (nmol, n)
    V = -(gam * tore.unsqueeze(1)).sum(-1)                              # (nmol, n)
    k = _norb(molecule, nbf)
    om = (torch.arange(nbf, device=M.device).view(1, 1, nbf) < k.unsqueeze(-1)).to(M.dtype)   # (nmol, n, nbf)
    add = torch.diag_embed(V.unsqueeze(-1) * om)                        # (nmol, n, nbf, nbf)
    blk = (torch.arange(nmol, device=M.device).view(-1, 1) * n * n + torch.arange(n, device=M.device).view(1, -1) * (n + 1))
    M = M.index_add(0, blk.flatten(), add.view(-1, nbf, nbf))
    G = gam.repeat_interleave(nbf, 1).repeat_interleave(nbf, 2)         # 軌道に展開（nmol, n·nbf, n·nbf）
    ctx.update(mol=molecule, gam=gam, far=far, om=om.view(nmol, n * nbf), G=G, nbf=nbf, n=n, npairs=int(far.sum()) // 2)
    return (M,) + tuple(out[1:])


def build_fock(problem, configuration, density):
    F = _ORIG_BUILD(problem, configuration, density)
    ctx = _get()
    if ctx is None or "gam" not in ctx:
        return F
    nmol = F.shape[0]; n, nbf = ctx["n"], ctx["nbf"]
    Pd = density.diagonal(dim1=1, dim2=2).reshape(nmol, n, nbf).sum(-1)          # N_B
    J = (ctx["gam"] * Pd.unsqueeze(1)).sum(-1)                                   # Σ_B γ_AB N_B
    Jo = J.repeat_interleave(nbf, 1) * ctx["om"]
    return F + torch.diag_embed(Jo) - 0.5 * ctx["G"] * density


def far_nuclear(Z, const, alp, chi, parameters, ccrep_mode):
    """遠い対の核間反発（eV、構造ごと）。式は近い対と同じ core._ccrep（feathering により γ は点電荷）"""
    ctx = _get()
    if ctx is None or "far" not in ctx:
        return None
    mol = ctx["mol"]; far = ctx["far"]
    nmol, n = far.shape[0], far.shape[1]
    real = (mol.species > 0).flatten()
    rank = torch.cumsum(real.long(), 0) - 1                               # (mol, atom) → PYSEQM の原子番号付け
    up = torch.triu(far, diagonal=1)                                      # a < b（Z_a ≥ Z_b の向き）
    m, a, b = up.nonzero(as_tuple=True)
    if m.numel() == 0:
        return None
    ii = rank[m * n + a]; jj = rank[m * n + b]
    x = mol.coordinates
    rij = (x[m, b] - x[m, a]).norm(dim=1) * mol.const.length_conversion_factor
    zero = torch.zeros_like(rij)
    E = core._ccrep(const, Z[ii], Z[jj], ii, jj, rij, zero, zero, alp, chi, parameters, pm7=(ccrep_mode == "PM7"))
    return torch.zeros(nmol, dtype=E.dtype, device=E.device).index_add(0, m, E)


def patch():
    global _ORIG_BUILD
    if _SL.hcore is not hcore:
        _SL.hcore = hcore
    if _ORIG_BUILD is None:
        _ORIG_BUILD = _SL.build_fock
        _SL.build_fock = build_fock
