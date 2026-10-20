"""結合リストと水素結合の組の探索を速くする（hbond.find_hbonds / setup_dh_plus、geom.set_up_dentate と同じ結果）

元の実装は Python の二重ループ（原子数の 2 乗）と、見つけた組の重複を全リストと比べる処理（組の数の 2 乗）を含み、
1674 原子の水クラスターで約 26 s かかっていた。ここでは距離の判定を numpy でまとめて行い、
順序に意味がある部分（重複の判定、結合数の上限、5 本目の結合を外す処理）だけを短い候補リストについて元と同じ順序で処理する。
"""
import math
import numpy as np
from . import hbond as HB
from . import geom as G

PI = math.pi


def _bangle_vec(X, k, j, i):
    """hbond._bangle_np(X, k, j, i) を i の配列について（スカラー版と同じ演算の順序）"""
    d2ij = ((X[k] - X[j]) ** 2).sum(-1); d2jk = ((X[j] - X[i]) ** 2).sum(-1); d2ik = ((X[k] - X[i]) ** 2).sum(-1)
    xy = np.sqrt(d2ij * d2jk)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.clip(0.5 * (d2ij + d2jk - d2ik) / xy, -1.0, 1.0)
    return np.where(xy < 1e-20, 0.0, np.arccos(c))


def find_hbonds(Z, X):
    Z = np.asarray(Z); X = np.asarray(X, float); n = len(Z)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    acc = np.nonzero((Z == 7) | (Z == 8))[0]
    Hs = np.nonzero(Z == 1)[0]
    if acc.size == 0 or Hs.size == 0:
        return [], D
    # hlist: 受容体の順 → 原子番号の順に、最初に 1.4 Å 以内に入った H を並べる（元の used の扱いと同じ）
    near = D[np.ix_(acc, Hs)] ** 2 < 1.4 ** 2                    # (nacc, nH)
    has = near.any(0)
    first = np.where(has, near.argmax(0), -1)
    order = np.lexsort((Hs[has], first[has]))
    hlist = Hs[has][order]
    pairs = []; seen = set()
    hl_set_pos = {int(h): p for p, h in enumerate(hlist)}
    for ia, i in enumerate(acc):
        js = hlist[(D[i, hlist] ** 2 < 1.4 ** 2)]                # hlist の順
        if js.size == 0:
            continue
        ks = acc[(acc != i) & (D[acc, i] ** 2 < 49.0)]           # acc の順
        if ks.size == 0:
            continue
        for j in js:
            ang = _bangle_vec(X, ks, j, i)
            for k, a in zip(ks, ang):
                if a > PI * 0.5:
                    if (int(k), int(j), int(i)) in seen:
                        continue
                    t = (int(i), int(j), int(k))
                    if t in seen:
                        continue
                    pairs.append(t); seen.add(t)
    return pairs, D


def setup_dh_plus(Z, X, pairs, D):
    Z = np.asarray(Z); n = len(Z)
    rad = HB.COVRAD[Z - 1]
    cache = {}

    def bonded_to(c):
        if c in cache:
            return cache[c]
        cand = np.nonzero((D[:, c] < rad + rad[c]) & (np.arange(n) != c))[0]
        lst = []
        for j in cand:                                          # 元のループと同じ順序で 5 本目の処理を再現
            lst.append(int(j))
            if len(lst) == 5:
                s = 0.0; l = 0
                for kk in range(5):
                    if s < D[c, lst[kk]]: s = D[c, lst[kk]]; l = kk
                lst.pop(l)
        cache[c] = lst
        return lst

    out = []
    Zl = [int(z) for z in Z]
    for (a1, h, a5) in pairs:
        hb = {1: a1, 9: h, 5: a5, 10: 0}
        bl1 = bonded_to(a1); bl2 = bonded_to(a5)
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
                hb[c4] = h if D[center, h] < HB._bonding(Zl, center, h) else center
            elif len(lst) == 1:
                hb[c2] = lst[0]
                c = hb[c2]
                l3 = [int(k) for k in np.nonzero((D[:, c] < rad + rad[c]) & (np.arange(n) != c))[0]]
                hb[c3] = far(l3)
                hb[c4] = h if D[center, h] < HB._bonding(Zl, center, h) else center
            elif len(lst) == 0:
                v = h if D[center, h] < HB._bonding(Zl, center, h) else center
                hb[c2] = hb[c3] = hb[c4] = v
        assign(a1, bl1, 2, 3, 4); assign(a5, bl2, 6, 7, 8)
        out.append(rec)
    return out


def set_up_dentate(Z, X):
    Z = np.asarray(Z); X = np.asarray(X, float); n = len(Z)
    rad = np.array([(-1.0 if G._T["is_metal"][z - 1] else G._T["atom_radius_covalent"][z - 1]) for z in Z])
    zi = Z[:, None]; zj = Z[None, :]
    il = np.minimum(zi, zj); iu = np.maximum(zi, zj)
    s = np.full((n, n), 1.1)
    s = np.where((il == 6) & (iu < 8), 1.2, s)
    s = np.where((il == 16) & (iu == 16), 1.2, s)
    s = np.where((il == 5) & (iu == 7), 1.0, s)
    s = np.where((il == 1) & (iu == 6), 1.25, s)
    r2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    cand = (r2 < (s * (rad[:, None] + rad[None, :])) ** 2) & np.tri(n, k=-1, dtype=bool)   # j < i
    nb = np.zeros(n, int); ib = [[] for _ in range(n)]
    ii, jj = np.nonzero(cand)                                  # i の昇順、同じ i では j の昇順（元のループと同じ）
    for i, j in zip(ii, jj):
        if nb[i] < 15 and nb[j] < 15:
            nb[i] += 1; nb[j] += 1; ib[i].append(int(j)); ib[j].append(int(i))
    for i in range(n):
        if Z[i] == 1 and nb[i] >= 2:
            ib[i] = [j for j in ib[i] if Z[j] != 1]; nb[i] = len(ib[i])
    return nb, ib
