"""PM7 の水素結合補正（MOPAC Hydrogen_bond_corrections + setup_DH_Plus + EH_plus の PM7 分岐）。kcal/mol
トポロジー（どの原子が供与体・受容体・置換基か）は MOPAC と同じく毎回の構造から決める（numpy）。
エネルギーは torch で書き、力は自動微分で得る（MOPAC は差分 δ=1e-5 Å）。"""
import math, numpy as np, torch
from .geom import _T, dist, bangle, dihed
PI = math.pi; DEG = PI / 180.0
A0 = 0.529177210903; EV = 27.211386245988; FPC9 = 23.060547830619029
HARTREE2KCAL = EV * FPC9
COVRAD = np.array(_T["hb_covrad"]) * 4.0 / 3.0
SC_NSP3 = -0.171271 * A0**2; SC_OSP3 = -0.098822 * A0**2; SC_NSP2 = SC_NSP3; SC_OSP2 = SC_OSP3

def _bonding(Z, a, b): return COVRAD[Z[a]-1] + COVRAD[Z[b]-1]

def _bangle_np(X, i, j, k):
    d2ij = ((X[i]-X[j])**2).sum(); d2jk = ((X[j]-X[k])**2).sum(); d2ik = ((X[i]-X[k])**2).sum()
    xy = math.sqrt(d2ij*d2jk)
    return 0.0 if xy < 1e-20 else math.acos(max(-1.0, min(1.0, 0.5*(d2ij+d2jk-d2ik)/xy)))

def find_hbonds(Z, X):
    """all_h_bonds（PM7: RAH=1.4 Å、cutoff=7 Å、受容体は N と O のみ）。戻り値: [(hb1, H, hb5)]"""
    n = len(Z); D = np.linalg.norm(X[:, None]-X[None], axis=-1)
    acc = [i for i in range(n) if Z[i] in (7, 8)]
    used = np.zeros(n, bool); hlist = []
    for i in acc:
        for j in range(n):
            if Z[j] == 1 and not used[j] and D[i, j]**2 < 1.4**2:
                hlist.append(j); used[j] = True
    pairs = []
    for i in acc:
        for j in hlist:
            if not D[i, j]**2 < 1.4**2: continue
            for k in acc:
                if k == i or not D[k, i]**2 < 49.0: continue
                if _bangle_np(X, k, j, i) > PI*0.5:
                    if any(p == (k, j, i) for p in pairs): continue
                    if any(p == (i, j, k) for p in pairs): continue
                    pairs.append((i, j, k))
    return pairs, D

def setup_dh_plus(Z, X, pairs, D):
    """setup_DH_Plus: hblist(1..10) と nrbondsa/nrbondsb を作る（1 始まりの列番号に合わせて dict で持つ）"""
    n = len(Z); out = []
    def bonded_to(c, D):
        lst = []
        for j in range(n):
            if D[j, c] < _bonding(Z, j, c) and j != c:
                lst.append(j)
                if len(lst) == 5:      # 多すぎる → 一番長い結合を外す
                    l = max(range(5), key=lambda kk: (D[c, lst[kk]], -kk)); 
                    # MOPAC: 最初の最大（sum < sum1 の厳密比較）を外す
                    best = -1; l = 0; s = 0.0
                    for kk in range(5):
                        if s < D[c, lst[kk]]: s = D[c, lst[kk]]; l = kk
                    lst.pop(l)
        return lst
    for (a1, h, a5) in pairs:
        hb = {1: a1, 9: h, 5: a5, 10: 0}
        bl1 = bonded_to(a1, D); bl2 = bonded_to(a5, D)
        if any(b == a5 for b in bl1): hb[10] = -666
        for b in bl1:
            for c in bl2:
                if b == c and c != h: hb[10] = -666
        rec = {"hb": hb, "na": len(bl1), "nb": len(bl2)}
        if hb[10] == -666: out.append(rec); continue
        def far(lst, excl=()):
            best, od = None, -1.0
            for b in lst:
                if b in excl: continue
                if D[b, h] > od: od = D[b, h]; best = b
            return best
        def assign(center, lst, c2, c3, c4):
            if len(lst) in (3, 4):
                hb[c2] = far(lst); hb[c3] = far(lst, (hb[c2],)); hb[c4] = far(lst, (hb[c2], hb[c3]))
            elif len(lst) == 2:
                hb[c2] = far(lst)
                for b in lst:
                    if b != hb[c2]: hb[c3] = b
                hb[c4] = h if D[center, h] < _bonding(Z, center, h) else center
            elif len(lst) == 1:
                hb[c2] = lst[0]
                l3 = [k for k in range(n) if D[k, hb[c2]] < _bonding(Z, k, hb[c2]) and k != hb[c2]]
                hb[c3] = far(l3)
                hb[c4] = h if D[center, h] < _bonding(Z, center, h) else center
            elif len(lst) == 0:
                v = h if D[center, h] < _bonding(Z, center, h) else center
                hb[c2] = hb[c3] = hb[c4] = v
        assign(a1, bl1, 2, 3, 4); assign(a5, bl2, 6, 7, 8)
        out.append(rec)
    return out

def _wrap(t):  # (−π, π] に
    t = torch.where(t <= -PI, t + 2*PI, t); return torch.where(t > PI, t - 2*PI, t)

def _side(Z, x, hb, nb, c1, c2, c3, c4, h):
    """EH_plus の片側（供与体側 c1=1 / 受容体側 c1=5）。戻り値: (angle2_cos, torsion_cos, ok) — ok=False なら 0 で打ち切り"""
    z = Z[hb[c1]]; ts = 0.0; a2s = 0.0; a2s2 = 0.0; tset = False; tset2 = False
    if z == 8:
        if nb == 1: a2s = PI; a2s2 = 120*DEG; ts = 0.0; tset2 = True
        else: a2s = 109.48*DEG; a2s2 = a2s; ts = 54.74*DEG
    elif z == 7:
        if nb == 2: a2s = 120*DEG; a2s2 = a2s; ts = 0.0
        else: a2s = 109.48*DEG; a2s2 = a2s; ts = 54.74*DEG; tset = True
    one = torch.tensor(1.0, dtype=x.dtype); zero = torch.tensor(0.0, dtype=x.dtype)
    ts = ts * one; a2s = a2s * one; a2s2 = a2s2 * one; tc = zero
    if tset:
        tc = _wrap(dihed(x, hb[c3], hb[c2], hb[c1], hb[c4]))
        tc = torch.where(tc < 0, -PI - tc, PI - tc)
        bac = tc; tcd = tc.abs() / DEG
        ts = ts + ((54.74 - tcd) / 54.74 * 35.26) * DEG
        a2s = a2s - ((54.74 - tcd) / 54.74 * 19.48) * DEG; a2s2 = a2s
        tc = bac
    ang2 = bangle(x, hb[c2], hb[c1], h)
    a2c = torch.maximum(torch.cos(a2s - ang2), torch.cos(a2s2 - ang2))
    if a2c.item() <= 0.0: return None
    tcor = _wrap(dihed(x, hb[c3], hb[c2], hb[c1], h))
    if (not tset2) or abs(tcor.item() / DEG) > 90.0:
        tcor = torch.where(tcor < 0, -PI - tcor, PI - tcor)
    tcv = tc.item()
    if tcv < 0: tcos = torch.cos(_wrap(ts - tcor))
    elif tcv > 0: tcos = torch.cos(_wrap(-ts - tcor))
    else: tcos = torch.maximum(torch.cos(_wrap(ts - tcor)), torch.cos(_wrap(-ts - tcor)))
    if dist(x, h, hb[c1]).item() > dist(x, h, hb[c2]).item() and tset2: tcos = zero
    if hb[c3] == hb[c4] or h == hb[c4]: tcos = one
    return a2c, tcos

def eh_plus(Z, x, rec):
    hb = rec["hb"]
    if hb[10] == -666: return None
    h = hb[9]
    angle_cos = -torch.cos(bangle(x, hb[1], h, hb[5]))
    if angle_cos.item() <= 0: return None
    s1 = _side(Z, x, hb, rec["na"], 1, 2, 3, 4, h)
    if s1 is None: return None
    a2c, tcos = s1
    if tcos.item() < 0: return None
    s2 = _side(Z, x, hb, rec["nb"], 5, 6, 7, 8, h)
    if s2 is None: return None
    a2cn, tcosn = s2; tcosn = tcosn.abs()
    def scale(c, nb):
        if Z[hb[c]] == 7: return SC_NSP3 if nb >= 3 else SC_NSP2
        return SC_OSP3 if nb >= 2 else SC_OSP2
    sc = 0.5 * (scale(1, rec["na"]) + scale(5, rec["nb"]))
    ha = dist(x, h, hb[1]); hbd = dist(x, h, hb[5]); xc = torch.minimum(ha, hbd)
    XY = torch.maximum(ha, hbd) - xc
    if XY.item() > 0.5: damping = 1.0 - 1.0 / (1.0 + torch.exp(-60.0 * (xc / 1.2 - 1.0)))
    else: damping = torch.ones_like(xc)
    xo = dist(x, hb[1], hb[5])
    damping = damping / (1.0 + torch.exp(-100.0 * (xo / 2.4 - 1.0)))
    damping = damping * (1.0 - 1.0 / (1.0 + torch.exp(-10.0 * (xo / 7.0 - 1.0))))
    E = sc / xo**2 * angle_cos**2 * (1.0 - (1.0 - a2c * tcos * a2cn * tcosn)**2) * HARTREE2KCAL * damping
    if Z[hb[1]] == 8 and Z[hb[5]] == 8:
        E = E - 2.5 * torch.exp(-80.0 * torch.clamp(xo - 2.67, min=0.0)**2) * angle_cos**4
    return E

def hbond_correction(Z, x, return_terms=False):
    X = x.detach().cpu().numpy()
    pairs, D = find_hbonds(Z, X)
    recs = setup_dh_plus(Z, X, pairs, D)
    tot = x.sum() * 0.0; terms = []
    for r in recs:
        e = eh_plus(Z, x, r)
        if e is not None:
            tot = tot + e; terms.append((r["hb"][1], r["hb"][9], r["hb"][5], float(e)))
    return (tot, terms) if return_terms else tot
