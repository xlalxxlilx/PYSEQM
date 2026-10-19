"""PYSEQM に MOPAC の PM7 固有の処理を差し込む（PYSEQM 本体のファイルは書き換えない）。

PM7 で MOPAC が PM6 と違う扱いをするのは次の 2 点（MOPAC 23.2.5 のソースで確認）:
 1. 核間反発項 ccrep: C–C 項の定数、O–H 追加項、二原子パラメータの無い組を同核の平均で埋める。
 2. 「feathering」（readmo.F90: l_feather = method_PM7）: 二中心積分・電子–芯引力・核間反発の γ を、
    r → 7 Å で点電荷に滑らかに切り替える（mndod.F90 の to_point、const = 1 − exp(−0.22 (r − 7)²)、r ≥ 7 Å で点電荷）。
"""
import contextlib, threading, torch
import seqm.basics as _B
from seqm.seqm_functions import energy as _EN
from seqm.seqm_functions import two_elec_two_center_int as _TE
from seqm.seqm_functions.constants import ev as _EV

VP7 = {1: 8.947612, 2: 6.024265, 3: -0.012037, 4: 0.701333}   # parameters_for_PM7_C.F90 の v_par7
VP6 = {1: 9.278465, 2: 5.983752}                               # parameters_for_PM6_C.F90 の v_par6
_A0 = 0.529167                                                  # PYSEQM の a0（PYSEQM 内部の換算と揃える）
TRUNC_1, TRUNC_2 = 7.0, 0.22                                    # run_mopac.F90 の既定値
_state = threading.local()
_orig_pne = _EN.pair_nuclear_energy
_orig_tetcilf = _TE.TETCILF

def _active():
    return getattr(_state, "mode", None)

@contextlib.contextmanager
def mode(m):
    """m = "PM7" / "PM6x"（PM6 の C–C 定数を MOPAC の値に）/ None（PYSEQM そのまま）"""
    old = _active(); _state.mode = m
    try: yield
    finally: _state.mode = old

def feather(r_bohr):
    """to_point: (const, point)。point は eV（ev/r）。"""
    rA = r_bohr * _A0
    const = torch.where(rA < TRUNC_1, 1.0 - torch.exp(-(rA - TRUNC_1) ** 2 * TRUNC_2), torch.zeros_like(rA))
    return const, _EV / r_bohr

_BLEND22 = [0, 2, 3, 10, 11, 15, 16, 17, 18, 20]               # MOPAC reppd: ri(1,3,4,11,12,16,17,18,19,21) は点電荷へ、他は ×const

def tetcilf(ni, nj, rij, *args, **kw):
    out = _orig_tetcilf(ni, nj, rij, *args, **kw)
    if _active() != "PM7": return out
    wHH, riXH, ri, coreHH, coreXH, core = out
    HH = (ni == 1) & (nj == 1); XH = (ni > 1) & (nj == 1); XX = (ni > 1) & (nj > 1)
    if wHH is not None and HH.any():
        c, p = feather(rij[HH]); wHH = wHH * c + (1 - c) * p
    if riXH is not None and XH.any():
        c, p = feather(rij[XH]); c = c[:, None]; p = p[:, None]
        m = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=riXH.dtype, device=riXH.device)
        riXH = riXH * c + (1 - c) * p * m
    if ri is not None and XX.any():
        c, p = feather(rij[XX]); c = c[:, None]; p = p[:, None]
        m = torch.zeros(22, dtype=ri.dtype, device=ri.device); m[_BLEND22] = 1.0
        ri = ri * c + (1 - c) * p * m
    return wHH, riXH, ri, coreHH, coreXH, core
_TE.TETCILF = tetcilf

def _ccrep(const, ni, nj, idxi, idxj, rij, rho0xi, rho0xj, alp, chi, parameters, pm7=True):
    """MOPAC ccrep（PM6/PM7 分岐）の移植。pm7=True で PM7 の定数・O–H 項・feathering。単位 eV"""
    VP = VP7 if pm7 else {**VP6, 3: 0.0, 4: 0.0}
    rA = rij * _A0; tore = const.tore; zz = tore[ni] * tore[nj]
    gab = _EV / torch.sqrt(rij * rij + (rho0xi + rho0xj) ** 2)
    if pm7:
        c, p = feather(rij); gab = gab * c + (1 - c) * p
    enuc = zz * gab
    fff = chi[ni, nj]; ab = alp[ni, nj]; ab = torch.where(ab < 1e-6, torch.full_like(ab, 1.2), ab)
    scale = 1.0 + 2.0 * fff * torch.exp(-ab * (rA + 0.0003 * rA ** 6))
    XH = ((ni == 6) | (ni == 7)) & (nj == 1); OH = (ni == 8) & (nj == 1)
    scale = torch.where(XH | OH, 1.0 + 2.0 * fff * torch.exp(-ab * rA ** 2), scale)
    scale = torch.where(OH, scale - VP[3] * torch.exp(-VP[4] * rA * 2), scale)
    scale = torch.where((ni == 6) & (nj == 6), scale + VP[1] * torch.exp(-VP[2] * rA), scale)
    scale = torch.where((ni == 14) & (nj == 8), scale - 0.7e-3 * torch.exp(-(rA - 2.9) ** 2), scale)
    has = fff.abs() > 1e-5
    generic = torch.abs(10.0 * torch.exp(-2.18 * rA) * enuc) + enuc               # 二原子パラメータの無い組（PM7 では平均で埋めるので通常使わない）
    E = torch.where(has, enuc * scale, generic)
    _, K, L, M = parameters
    v = 0.0
    for X in (idxi, idxj):                                                         # PM6/PM7 は第 1 Gaussian のみ
        ax = L[X, 0] * (rA - M[X, 0]) ** 2
        v = v + torch.where(ax < 25.0, zz / rA * K[X, 0] * torch.exp(-ax), torch.zeros_like(rA))
    E = E + v
    ax = rA / (ni.to(rA.dtype) ** 0.3333 + nj.to(rA.dtype) ** 0.3333)
    E = E + torch.where(ax < 3.0, torch.clamp(1e-8 / ax ** 12, max=1e5), torch.zeros_like(ax))
    return E

def pair_nuclear_energy(Z, const, nmol, ni, nj, idxi, idxj, rij, rho0xi, rho0xj, alp, chi, gam, method="AM1", parameters=None):
    m = _active()
    if m in ("PM7", "PM6x") and method in ("PM6", "PM6_SP"):
        E = _ccrep(const, ni, nj, idxi, idxj, rij, rho0xi, rho0xj, alp, chi, parameters, pm7=(m == "PM7"))
        from . import farfield
        Ef = farfield.far_nuclear(Z, const, alp, chi, parameters, m)
        if Ef is not None:                                                    # 打ち切った遠い対の分を各構造の最初の対に足す
            pm = farfield._get()["mol"].pair_molid
            cnt = torch.bincount(pm, minlength=Ef.shape[0])
            if bool((cnt == 0).any()):
                raise ValueError("打ち切り距離の内側に原子対が 1 つも無い構造がある（cutoff を大きくする）")
            first = torch.zeros(Ef.shape[0], dtype=torch.long, device=E.device).scatter_reduce(
                0, pm, torch.arange(pm.numel(), device=E.device), reduce="amin", include_self=False)
            E = E.index_add(0, first, Ef)
        return E
    E = _orig_pne(Z, const, nmol, ni, nj, idxi, idxj, rij, rho0xi, rho0xj, alp, chi, gam, method=method, parameters=parameters)
    if m == "PM6x" and method in ("PM6", "PM6_SP"):
        rija = rij * _A0; zz = const.tore[ni] * const.tore[nj]
        gab = _EV / torch.sqrt(rij * rij + (rho0xi + rho0xj) ** 2)
        XCC = (ni == 6) & (nj == 6)
        E = E + torch.where(XCC, zz * gab * (VP6[1] * torch.exp(-VP6[2] * rija) - 9.28 * torch.exp(-5.98 * rija)), torch.zeros_like(E))
    return E
_B.pair_nuclear_energy = pair_nuclear_energy
