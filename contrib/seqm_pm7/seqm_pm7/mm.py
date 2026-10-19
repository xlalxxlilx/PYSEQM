"""MOPAC の MM 型補正（C≡C 三重結合、アミド NHCO 二面角）。エネルギーは kcal/mol"""
import numpy as np, torch
from .geom import dist, dihed

def c_triple_bond_c(Z, x, ibonds):
    """C_triple_bond_C（PM6/PM7）。ibonds は set_up_dentate の結合リスト（MOPAC は入力時の構造で固定）"""
    rmin, rmax, p1, p2 = 1.21, 1.33, -5.0, 25.0
    pairs = [(i, j) for i in range(len(Z)) if Z[i] == 6 for j in ibonds[i] if j < i and Z[j] == 6]
    if not pairs: return x.sum() * 0.0
    a = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])
    r = dist(x, a, b)
    t = ((r - rmin) / (rmax - rmin)).clamp(0.0, 1.0)
    sw = 1.0 - 10 * t**3 + 15 * t**4 - 6 * t**5 + (p1 + t * p2) * (t**3 - 3 * t**4 + 3 * t**5 - t**6)
    val = torch.where(r < rmin, torch.ones_like(r), torch.where(r < rmax, sw, torch.zeros_like(r)))
    return 12.0 * val.sum()

HTYPE = {"PM6": 2.5000, "PM7": 3.1595}

def setup_nhco(Z, X):
    """MOPAC setup_nhco: O=C–N(–H)–X の組を探す（毎回の構造で判定）。X は numpy"""
    n = len(Z); D = np.linalg.norm(X[:, None] - X[None], axis=-1); out = []
    for j in range(n):
        if Z[j] != 6: continue
        found = False
        for i in range(n):
            if found or Z[i] != 8 or D[i, j] > 1.3: continue
            for k in range(n):
                if found or Z[k] != 7 or D[k, j] > 1.6: continue
                for l in range(n):
                    if found or Z[l] != 1 or D[k, l] > 1.3: continue
                    for m in range(n):
                        if m in (k, l, j) or D[m, k] > 1.7: continue
                        if any(out[jj][2] == k for jj in range(0, len(out), 2)): continue
                        out.append((i, j, k, m)); out.append((i, j, k, l)); found = True; break
    return out

def nhco_correction(Z, x, method):
    X = x.detach().cpu().numpy(); quads = setup_nhco(Z, X)
    if not quads: return x.sum() * 0.0, 0
    q = torch.tensor(quads)
    ang = dihed(x, q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    return HTYPE[method] * (torch.sin(ang) ** 2).sum(), len(quads)

def nsp2_correction(Z, x, ibonds):
    """PM6 のみ: 3 結合の N（H が 1 個以下）のピラミッド化補正。ibonds は入力時の構造で固定"""
    terms = []
    for n in range(len(Z)):
        if Z[n] == 7 and len(ibonds[n]) == 3:
            i, j, k = ibonds[n]
            if sum(Z[a] == 1 for a in (i, j, k)) < 2:
                terms.append((n, i, j, k))
    if not terms: return x.sum() * 0.0
    t = torch.tensor(terms); n, i, j, k = t.unbind(1)
    a = dist(x, n, i); b = dist(x, n, j); c = dist(x, n, k); ab = dist(x, j, i); ac = dist(x, k, i); bc = dist(x, j, k)
    acos = lambda v: torch.acos(v.clamp(-1 + 1e-15, 1 - 1e-15))
    ca = acos((b**2 + c**2 - bc**2) / (2 * b * c)); cb = acos((a**2 + c**2 - ac**2) / (2 * a * c)); cc = acos((b**2 + a**2 - ab**2) / (2 * b * a))
    tot = 2 * np.pi - (ca + cb + cc)
    return (-0.5 * torch.exp(-10.0 * tot)).sum()

def si_o_h_correction(Z, x, ibonds):
    """Si–O–H 補正（PM7 のみ、MOPAC set_up_dentate.F90 の Si_O_H_Correction、kcal/mol）。
    結合リストで Si と H の両方に結合した O ごとに（Si・H が複数なら番号の大きい方）
    15·(∠Si–O–H − 125°)²·exp(−33·max(0, r²_SiO − 1.7²))·exp(−68·max(0, r²_OH − 1))（角度はラジアン、r は Å）"""
    import math
    trip = []
    for o in range(len(Z)):
        if Z[o] != 8: continue
        si = h = None
        for k in ibonds[o]:
            if Z[k] == 14: si = k
            if Z[k] == 1: h = k
        if si is not None and h is not None: trip.append((si, o, h))
    if not trip: return x.sum() * 0.0
    from .geom import bangle
    tot = x.sum() * 0.0
    for si, o, h in trip:
        rso = ((x[o] - x[si])**2).sum() - 1.7**2; roh = ((x[o] - x[h])**2).sum() - 1.0
        ang = bangle(x, si, o, h)
        tot = tot + 15.0 * (ang - 125.0 * math.pi / 180.0)**2 * torch.exp(-33.0 * rso.clamp_min(0.0)) * torch.exp(-68.0 * roh.clamp_min(0.0))
    return tot
