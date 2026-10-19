"""PM7 / PM6（MOPAC 互換の補正つき）を PYSEQM で計算する入口"""
import io, os, contextlib, numpy as np, torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.ElectronicStructure import Electronic_Structure
from . import core, constants_switch
from .build_params import build, build_pm6
from .geom import set_up_dentate
from .mm import c_triple_bond_c, nhco_correction, nsp2_correction, si_o_h_correction
from .dispersion import dispersion
from .hbond_vec import hbond_correction
from . import tables
from ase.data import chemical_symbols as _SYMS
import csv
tables.patch_constants(); tables.patch_rotate()
from . import overlap_patch; overlap_patch.patch()
from . import scf_fast; scf_fast.patch()
from . import fock_fast; fock_fast.patch()
from . import rot_fast; rot_fast.patch()
from . import farfield; farfield.patch()
EV2KCAL = 23.060547830619029
SYM = list(_SYMS[1:87])
torch.set_default_dtype(torch.float64)
PM7_DIR = build()
PM7_D = set(int(z) for z in open(PM7_DIR + "d_elements.txt").read().split())
PM6_DIR = build_pm6()
PM6_D = set(int(z) for z in open(PM6_DIR + "d_elements.txt").read().split())
_PCACHE = {}
def _params(pdir):
    if pdir not in _PCACHE:
        _PCACHE[pdir] = {int(r["N"]): {k.strip(): float(v) for k, v in r.items() if k.strip() not in ("N", "sym")}
                         for r in csv.DictReader(open(pdir + "parameters_PM6_MOPAC.csv"))}
    return _PCACHE[pdir]

def _Z(symbols):
    return np.array([s if isinstance(s, (int, np.integer)) else SYM.index(s) + 1 for s in symbols])

class Topology:
    """MOPAC が入力時に一度だけ作る結合リスト（C≡C、nsp2、分散の炭素 C6 に使う）"""
    def __init__(self, Z, X): self.nbonds, self.ibonds = set_up_dentate(list(Z), np.asarray(X))

def compute(symbols, coords, method="PM7", charge=0, device="cpu", mm=True, disp=None, hbond=None,
            topology=None, scf_eps=1e-9, P0=None, return_density=False, constants="codata", scf_converger=None, pre_iter=8, pre_mode="inline", cutoff=None, cutoff_approx=False, _return_mol=False):
    """symbols: 元素記号か原子番号の列（全分子で共通）。coords: (n,3) または (nmol,n,3) Å。
    method: "PM7"（MOPAC の PM7 に相当）、"PM7-"（補正なし = MOPAC の PM7-）、"PM6"（MOPAC の PM6 に相当）。
    戻り値は kcal/mol と kcal/mol/Å。"""
    if constants_switch.current() != constants: constants_switch.use(constants)
    cv = constants_switch.CODATA if constants == "codata" else constants_switch.OLD
    fpc9 = 23.060547830619029 if constants == "codata" else 23.061
    Z = _Z(symbols); X = np.asarray(coords, float); single = X.ndim == 2
    if single: X = X[None]
    nmol, n, _ = X.shape
    m = method.upper()
    if m in ("PM7", "PM7-"):
        disp = (m == "PM7") if disp is None else disp; hbond = (m == "PM7") if hbond is None else hbond
        smeth, pdir, cmode, mmname = ("PM6" if set(Z) & PM7_D else "PM6_SP"), PM7_DIR, "PM7", "PM7"
    elif m == "PM6":
        disp = False if disp is None else disp; hbond = False if hbond is None else hbond
        smeth, pdir, cmode, mmname = ("PM6" if set(Z) & PM6_D else "PM6_SP"), PM6_DIR, "PM6x", "PM6"
    else:
        raise ValueError(method)
    if cutoff is not None:
        # 距離による打ち切り（farfield）。PM7 では R ≥ 7 Å の積分が点電荷なので、R_c ≥ 7 Å ならほぼ厳密
        if float(cutoff) < 7.0:
            raise ValueError("cutoff は 7 Å 以上にする（PM7 の feathering が点電荷になる距離）")
        if m != "PM7" and m != "PM7-" and not cutoff_approx:
            raise ValueError("PM6 には feathering が無く、打ち切りは近似になる。使うなら cutoff_approx=True")
    o = np.argsort(-Z, kind="stable")
    dev = torch.device(device)
    species = torch.as_tensor(np.tile(Z[o], (nmol, 1)), device=dev)
    xyz = torch.as_tensor(X[:, o], device=dev)
    if scf_converger is None:
        # d 軌道をもつ系では PYSEQM の Pulay DIIS（[2]）が誤った SCF 解に落ちることがあり（例: PM6 の ZrH4）、
        # 適応混合（[1]）は収束しないことがある（例: PM7 の HfMe4）。収束するまで順に試す
        # 初期密度 P0 が与えられたとき（構造最適化・MD の前ステップの密度など）は前処理を省いて DIIS から始める
        convs = ([("pre", 2, 0.0), (1,), (0, 0.1)] if P0 is None else [(2, 0.0), (1,), (0, 0.1)]) if smeth == "PM6" else [(2, 0.0), (1,), (0, 0.1)]
    else:
        convs = [tuple(scf_converger)]
    par = {"method": smeth, "scf_eps": scf_eps, "scf_converger": list(convs[0]), "sp2": [False], "pair_outer_cutoff": (float(cutoff) if cutoff is not None else 1e10),
           "eig": True, "analytical_gradient": [False]}
    if pdir: par["parameter_file_dir"] = pdir
    cv_a0 = cv["a0"]
    const = Constants(length_conversion_factor=1.0 / cv["a0"]).to(dev)
    ch = torch.as_tensor(np.broadcast_to(np.array(charge, float), (nmol,)).copy(), device=dev)
    import warnings
    import seqm.seqm_functions.scf_loop as _SL
    from seqm.basics import Energy as _Energy
    for ic, cv in enumerate(convs):
        P_start = P0
        with core.mode(cmode), farfield.active(cutoff), contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            if ic < len(convs) - 1: warnings.simplefilter("ignore")
            hyb = 0
            if cv[0] == "pre" and pre_mode == "inline":
                # 適応混合を最大 pre_iter 回 → 同じ積分のまま DIIS（scf_fast）
                cv = tuple(cv[1:]); hyb = pre_iter
            elif cv[0] == "pre":
                # （旧方式）適応混合を別の計算として pre_iter 回だけ回してから DIIS に切り替える
                cv = tuple(cv[1:])
                par["scf_converger"] = [1]
                old_max = _SL.MAX_ITER; _SL.MAX_ITER = pre_iter
                try:
                    with torch.no_grad():
                        m0 = Molecule(const, par, xyz.detach(), species, charges=ch).to(dev)
                        P_start = _Energy(par).to(dev)(m0, P0=P0, do_force=False)[2].detach()
                finally:
                    _SL.MAX_ITER = old_max
            par["scf_converger"] = list(cv)
            mol = Molecule(const, par, xyz, species, charges=ch).to(dev)
            es = Electronic_Structure(par).to(dev)
            with scf_fast.hybrid(hyb):
                es(mol, P0=P_start)
        nc = getattr(es, "notconverged", None)
        converged = nc is None or not bool(torch.as_tensor(nc).any())
        if converged: break
    eheat_ev = const.eheat[species].sum(1).detach().cpu().numpy()          # PYSEQM は kcal/mol ÷ 23.061 で保持
    Hscf = (mol.Hf.detach().cpu().numpy() - eheat_ev) * fpc9 + eheat_ev * 23.061
    # PYSEQM の孤立原子エネルギーに無い d 項（MOPAC の udd*iod + eiscor）を差し引く
    if smeth == "PM6":
        P = _params(pdir)
        Hscf = Hscf - sum(tables.eisol_d(int(z), P[int(z)]) for z in Z) * fpc9
    F = np.zeros((nmol, n, 3)); F[:, o] = mol.force.detach().cpu().numpy() * fpc9
    comp = {k: np.zeros(nmol) for k in ("E_CC", "E_NHCO", "E_NSP2", "E_SiOH", "E_disp", "E_hb")}
    topo = topology or Topology(Z, X[0])
    Zl = list(int(z) for z in Z)
    for b in range(nmol):
        x = torch.tensor(X[b], requires_grad=True)
        terms = {}
        if mm:
            terms["E_CC"] = c_triple_bond_c(Zl, x, topo.ibonds)
            terms["E_NHCO"] = nhco_correction(Zl, x, mmname)[0]
            if mmname == "PM6": terms["E_NSP2"] = nsp2_correction(Zl, x, topo.ibonds)
            if mmname == "PM7" and 14 in Zl: terms["E_SiOH"] = si_o_h_correction(Zl, x, topo.ibonds)
        if disp: terms["E_disp"] = dispersion(Zl, x, topo.nbonds)
        if hbond: terms["E_hb"] = hbond_correction(Zl, x)
        if terms:
            tot = sum(terms.values())
            g, = torch.autograd.grad(tot, x, allow_unused=True)
            if g is not None: F[b] -= g.numpy()
            for k, v in terms.items(): comp[k][b] = float(v)
    Htot = Hscf + sum(comp.values())
    out = {"Hf": Htot, "Hf_scf": Hscf, "forces": F, **comp, "topology": topo, "scf_converged": converged, "scf_converger": tuple(cv)}
    if return_density: out["P"] = mol.dm
    if _return_mol: out["_mol"] = mol
    if single: out = {k: (v[0] if isinstance(v, np.ndarray) and k != "forces" else (v[0] if k == "forces" else v)) for k, v in out.items()}
    return out
