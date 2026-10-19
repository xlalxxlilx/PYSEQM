"""hbond.py の EH_plus を全ペア一括（torch ベクトル化）で評価する版。トポロジーは hbond.py と同じ numpy 実装を使う。"""
import math, numpy as np, torch
from .hbond import find_hbonds, setup_dh_plus, PI, DEG, HARTREE2KCAL, SC_NSP3, SC_NSP2, SC_OSP3, SC_OSP2

def _wrap(t):
    t = torch.where(t <= -PI, t + 2*PI, t); return torch.where(t > PI, t - 2*PI, t)

def _safe_dihed(x, i, j, k, l):
    """geom.dihed と同じ定義。退化（s=c=0）では 0 を返し、勾配が NaN にならないようにする"""
    vi = x[i] - x[k]; vj = x[j] - x[k]; vl = x[l] - x[k]
    xi1, yi1, zi1 = vi.unbind(-1); xj1, yj1, zj1 = vj.unbind(-1); xl1, yl1, zl1 = vl.unbind(-1)
    d2 = (vj ** 2).sum(-1); okd = d2 > 1e-20
    d = torch.sqrt(torch.where(okd, d2, torch.ones_like(d2)))
    cosa = torch.where(okd, zj1 / d, torch.ones_like(d)).clamp(-1.0, 1.0)
    s2 = (1.0 - cosa ** 2); oks = s2 > 1e-24
    yx = d * torch.sqrt(torch.where(oks, s2, torch.ones_like(s2)))
    ok = oks & (yx > 1e-6)
    yxs = torch.where(ok, yx, torch.ones_like(yx))
    cosph = torch.where(ok, yj1 / yxs, torch.ones_like(yx)); sinph = torch.where(ok, xj1 / yxs, torch.zeros_like(yx))
    xi2 = xi1 * cosph - yi1 * sinph; xl2 = xl1 * cosph - yl1 * sinph
    yi2 = xi1 * sinph + yi1 * cosph; yj2 = xj1 * sinph + yj1 * cosph; yl2 = xl1 * sinph + yl1 * cosph
    sinth = torch.where(ok, yj2 / d, torch.zeros_like(d)); costh = cosa
    yi3 = yi2 * costh - zi1 * sinth; yl3 = yl2 * costh - zl1 * sinth
    s = xl2 * yi3 - yl3 * xi2; c = xl2 * xi2 + yl3 * yi3
    deg = (s * s + c * c) < 1e-24
    ang = torch.atan2(torch.where(deg, torch.zeros_like(s), s), torch.where(deg, torch.ones_like(c), c))
    return torch.remainder(ang, 2 * PI)

def _safe_bangle(x, i, j, k):
    d2ij = ((x[i] - x[j]) ** 2).sum(-1); d2jk = ((x[j] - x[k]) ** 2).sum(-1); d2ik = ((x[i] - x[k]) ** 2).sum(-1)
    p = d2ij * d2jk; ok = p > 1e-40
    xy = torch.sqrt(torch.where(ok, p, torch.ones_like(p)))
    t = torch.where(ok, 0.5 * (d2ij + d2jk - d2ik) / xy, torch.ones_like(p)).clamp(-1.0 + 1e-15, 1.0 - 1e-15)
    return torch.where(ok, torch.acos(t), torch.zeros_like(t))

def _dist(x, a, b):
    d2 = ((x[a] - x[b]) ** 2).sum(-1); return torch.sqrt(torch.where(d2 > 1e-24, d2, torch.ones_like(d2)))

def _side(Zt, x, H, C1, C2, C3, C4, nb):
    z = Zt[C1]; f = x.dtype
    isO = z == 8; isN = z == 7
    o1 = isO & (nb == 1); o2 = isO & ~(nb == 1); n2 = isN & (nb == 2); n3 = isN & ~(nb == 2)
    T = lambda v: torch.full(C1.shape, v, dtype=f)
    a2s = torch.where(o1, T(PI), torch.where(o2 | n3, T(109.48*DEG), torch.where(n2, T(120*DEG), T(0.0))))
    a2s2 = torch.where(o1, T(120*DEG), a2s)
    ts = torch.where(o2 | n3, T(54.74*DEG), T(0.0))
    tset = n3; tset2 = o1
    tc = _wrap(_safe_dihed(x, C3, C2, C1, C4)); tc = torch.where(tc < 0, -PI - tc, PI - tc)
    tcd = tc.abs() / DEG
    ts = torch.where(tset, ts + ((54.74 - tcd) / 54.74 * 35.26) * DEG, ts)
    a2s = torch.where(tset, a2s - ((54.74 - tcd) / 54.74 * 19.48) * DEG, a2s); a2s2 = torch.where(tset, a2s, a2s2)
    tc = torch.where(tset, tc, torch.zeros_like(tc))
    ang2 = _safe_bangle(x, C2, C1, H)
    a2c = torch.maximum(torch.cos(a2s - ang2), torch.cos(a2s2 - ang2))
    tcor = _wrap(_safe_dihed(x, C3, C2, C1, H))
    flip = (~tset2) | (tcor.detach().abs() / DEG > 90.0)
    tcor = torch.where(flip, torch.where(tcor < 0, -PI - tcor, PI - tcor), tcor)
    c_neg = torch.cos(_wrap(ts - tcor)); c_pos = torch.cos(_wrap(-ts - tcor))
    tcos = torch.where(tc < 0, c_neg, torch.where(tc > 0, c_pos, torch.maximum(c_neg, c_pos)))
    farther = _dist(x, H, C1).detach() > _dist(x, H, C2).detach()
    tcos = torch.where(tset2 & farther, torch.zeros_like(tcos), tcos)
    tcos = torch.where((C3 == C4) | (H == C4), torch.ones_like(tcos), tcos)
    return a2c, tcos

def hbond_correction(Z, x, return_terms=False):
    X = x.detach().cpu().numpy()
    pairs, D = find_hbonds(Z, X)
    recs = [r for r in setup_dh_plus(Z, X, pairs, D) if r["hb"][10] != -666]
    if not recs: return (x.sum() * 0.0, []) if return_terms else x.sum() * 0.0
    col = lambda c: torch.tensor([r["hb"][c] for r in recs])
    A1, A2, A3, A4, A5, A6, A7, A8, H = (col(c) for c in (1, 2, 3, 4, 5, 6, 7, 8, 9))
    na = torch.tensor([r["na"] for r in recs]); nb = torch.tensor([r["nb"] for r in recs])
    Zt = torch.tensor(Z)
    angle_cos = -torch.cos(_safe_bangle(x, A1, H, A5))
    a2c, tcos = _side(Zt, x, H, A1, A2, A3, A4, na)
    a2cn, tcosn = _side(Zt, x, H, A5, A6, A7, A8, nb); tcosn = tcosn.abs()
    valid = (angle_cos > 0) & (a2c > 0) & (tcos >= 0) & (a2cn > 0)
    def scale(C, n):
        return torch.where(Zt[C] == 7, torch.where(n >= 3, torch.tensor(SC_NSP3), torch.tensor(SC_NSP2)),
                           torch.where(n >= 2, torch.tensor(SC_OSP3), torch.tensor(SC_OSP2))).to(x.dtype)
    sc = 0.5 * (scale(A1, na) + scale(A5, nb))
    ha = _dist(x, H, A1); hbd = _dist(x, H, A5); xc = torch.minimum(ha, hbd); XY = torch.maximum(ha, hbd) - xc
    damping = torch.where(XY.detach() > 0.5, 1.0 - 1.0 / (1.0 + torch.exp(-60.0 * (xc / 1.2 - 1.0))), torch.ones_like(xc))
    xo = _dist(x, A1, A5)
    damping = damping / (1.0 + torch.exp(-100.0 * (xo / 2.4 - 1.0)))
    damping = damping * (1.0 - 1.0 / (1.0 + torch.exp(-10.0 * (xo / 7.0 - 1.0))))
    E = sc / xo ** 2 * angle_cos ** 2 * (1.0 - (1.0 - a2c * tcos * a2cn * tcosn) ** 2) * HARTREE2KCAL * damping
    OO = (Zt[A1] == 8) & (Zt[A5] == 8)
    E = E + torch.where(OO, -2.5 * torch.exp(-80.0 * torch.clamp(xo - 2.67, min=0.0) ** 2) * angle_cos ** 4, torch.zeros_like(E))
    E = torch.where(valid, E, torch.zeros_like(E))
    tot = E.sum()
    if return_terms:
        return tot, [(int(a), int(h), int(b), float(e)) for a, h, b, e in zip(A1, H, A5, E.detach()) if abs(float(e)) > 0]
    return tot
