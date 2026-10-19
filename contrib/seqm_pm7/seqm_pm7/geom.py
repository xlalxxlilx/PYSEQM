"""MOPAC と同じ定義の幾何量（torch、微分可能）と結合トポロジー（numpy）"""
import json, os, math, numpy as np, torch
_T = json.load(open(os.path.join(os.path.dirname(__file__), "data_tables.json")))
TWO_PI = 2.0 * math.pi

def dist(x, a, b):
    return torch.linalg.vector_norm(x[a] - x[b], dim=-1)

def bangle(x, i, j, k):
    """MOPAC bangle: j を頂点とする角 i-j-k（ラジアン, 0..π）"""
    d2ij = ((x[i] - x[j]) ** 2).sum(-1); d2jk = ((x[j] - x[k]) ** 2).sum(-1); d2ik = ((x[i] - x[k]) ** 2).sum(-1)
    xy = torch.sqrt(d2ij * d2jk).clamp_min(1e-20)
    t = (0.5 * (d2ij + d2jk - d2ik) / xy).clamp(-1.0 + 1e-15, 1.0 - 1e-15)
    return torch.acos(t)

def dihed(x, i, j, k, l):
    """MOPAC dihed の移植（k を原点、j−k を z 軸にとる回転）。戻り値は [0, 2π)"""
    vi = x[i] - x[k]; vj = x[j] - x[k]; vl = x[l] - x[k]
    xi1, yi1, zi1 = vi.unbind(-1); xj1, yj1, zj1 = vj.unbind(-1); xl1, yl1, zl1 = vl.unbind(-1)
    d = torch.linalg.vector_norm(vj, dim=-1)
    cosa = (zj1 / d).clamp(-1.0, 1.0)
    yx = d * torch.sqrt((1.0 - cosa ** 2).clamp_min(0.0))
    ok = yx > 1e-6
    yxs = torch.where(ok, yx, torch.ones_like(yx))
    cosph = torch.where(ok, yj1 / yxs, torch.ones_like(yx)); sinph = torch.where(ok, xj1 / yxs, torch.zeros_like(yx))
    xi2 = xi1 * cosph - yi1 * sinph; xl2 = xl1 * cosph - yl1 * sinph
    yi2 = xi1 * sinph + yi1 * cosph; yj2 = xj1 * sinph + yj1 * cosph; yl2 = xl1 * sinph + yl1 * cosph
    sinth = torch.where(ok, yj2 / d, torch.zeros_like(d)); costh = cosa
    yi3 = yi2 * costh - zi1 * sinth; yl3 = yl2 * costh - zl1 * sinth
    # dang(xl2, yl3, xi2, yi3): a=(xl2,yl3), b=(xi2,yi3) のなす角。符号つき → [0,2π)
    s = xl2 * yi3 - yl3 * xi2; c = xl2 * xi2 + yl3 * yi3
    ang = torch.atan2(s, c)
    return torch.remainder(ang, TWO_PI)

def set_up_dentate(Z, X):
    """MOPAC set_up_dentate: 結合リスト（ibonds）と結合数（nbonds）。X は numpy (n,3)、Z は原子番号"""
    n = len(Z); rad = np.array([(-1.0 if _T["is_metal"][z - 1] else _T["atom_radius_covalent"][z - 1]) for z in Z])
    nb = np.zeros(n, int); ib = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            r2 = ((X[i] - X[j]) ** 2).sum(); il, iu = min(Z[i], Z[j]), max(Z[i], Z[j]); s = 1.1
            if il == 1 and iu == 6: s = 1.25
            elif il == 5 and iu == 7: s = 1.0
            elif il == 6 and iu < 8: s = 1.2
            elif il == 16 and iu == 16: s = 1.2
            if r2 < (s * (rad[i] + rad[j])) ** 2 and nb[i] < 15 and nb[j] < 15:
                nb[i] += 1; nb[j] += 1; ib[i].append(j); ib[j].append(i)
    for i in range(n):  # 2 本以上の結合をもつ H からは H–H 結合を外す
        if Z[i] == 1 and nb[i] >= 2:
            ib[i] = [j for j in ib[i] if Z[j] != 1]; nb[i] = len(ib[i])
    return nb, ib
