"""PYSEQM の diatom_overlap_matrixD（d 軌道を含む系の重なり積分）を sto_overlap.overlap_molecular に置き換える。
PYSEQM の実装は (n, l) の組ごとの展開式の一部（5d–1s, 4d–4d, 5p–3p, 6s–3s など）と、d–d の組の回転に誤りがある。
置き換え後は局所系の積分も回転も自前（pyscf の数値格子積分と 1e-12 で一致）。
下の local_integrals / _hook は、PYSEQM の局所系の値と比べるための診断用（patch() では使わない）。"""
import inspect, textwrap, torch
from .sto_overlap import overlap_local

NAMES = ["S111", "S211", "S121", "S221", "S222", "S311", "S321", "S322", "S131", "S231", "S232", "S331", "S332", "S333"]
# PYSEQM の局所系の符号（sto_overlap の規約に対する比）。正しく計算される組（Zr–O, Si–Si など）で決めた値
SIGN = {"S121": -1.0, "S221": -1.0, "S321": -1.0, "S232": -1.0}

def local_integrals(ni, nj, rij, zeta_a, zeta_b, qn_int, qnD_int):
    out = {}
    na_sp, nb_sp = qn_int[ni], qn_int[nj]; na_d, nb_d = qnD_int[ni], qnD_int[nj]
    for name in NAMES:
        la, lb, m = int(name[1]) - 1, int(name[2]) - 1, int(name[3]) - 1
        na = na_d if la == 2 else na_sp; nb = nb_d if lb == 2 else nb_sp
        za, zb = zeta_a[..., la], zeta_b[..., lb]
        ok = (za > 0) & (zb > 0) & (na > 0) & (nb > 0)
        S = torch.zeros_like(rij)
        if ok.any():
            keys = torch.stack([na, nb], -1)[ok]
            for key in torch.unique(keys, dim=0).tolist():
                sel = ok & (na == key[0]) & (nb == key[1])
                S = S.index_put((sel.nonzero(as_tuple=True)[0],),
                                SIGN.get(name, 1.0) * overlap_local(key[0], la, key[1], lb, m, za[sel], zb[sel], rij[sel]))
        out[name] = S
    return out

_capture = None
def _hook(ni, nj, rij, zeta_a, zeta_b, qn_int, qnD_int, loc):
    """差し込んだ行から呼ばれる。_capture が dict のときは PYSEQM 自身の値を記録するだけ（符号の較正用）"""
    if isinstance(_capture, dict):
        _capture.update({k: loc[k].detach().clone() for k in NAMES}); return [loc[k] for k in NAMES]
    new = local_integrals(ni, nj, rij, zeta_a, zeta_b, qn_int, qnD_int)
    return [new[k] for k in NAMES]

_patched = False
def patch():
    """PYSEQM の diatom_overlap_matrixD を sto_overlap.overlap_molecular（局所積分も回転も自前）に置き換える"""
    global _patched
    if _patched: return
    import seqm.seqm_functions.diat_overlapD as D, seqm.seqm_functions.hcore as H, seqm.seqm_functions.tools as T
    from .sto_overlap import overlap_molecular
    D._pyseqm_diatom_overlap_matrixD = D.diatom_overlap_matrixD
    D.diatom_overlap_matrixD = overlap_molecular
    H.diatom_overlap_matrixD = overlap_molecular; T.diatom_overlap_matrixD = overlap_molecular
    _patched = True
