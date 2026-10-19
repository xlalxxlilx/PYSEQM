"""任意の主量子数の Slater 型軌道（s, p, d）の二中心重なり積分（局所座標系、微分可能）

PYSEQM の diat_overlapD.py は (n, l) の組ごとに展開式を手で書き下しているが、5d（Hf など）や
4d–4d、5p–3p などの組で誤った値を返す（数値積分との比較で確認。例: Zr–Zr の dz2–s が 1.64）。
ここでは被積分関数を回転楕円体座標 ξ, η で多項式に展開し（sympy、組ごとに一度だけ）、
補助積分 A_k(p) = ∫_1^∞ ξ^k e^{-pξ} dξ、B_k(t) = ∫_{-1}^{1} η^k e^{-tη} dη の線形結合として評価する。

規約: 原子 a を原点、原子 b を +z 上（距離 R bohr）に置き、両原子とも同じ向きの z 軸を使う。
角度部分は実球面調和関数（s; pσ=pz, pπ=px; dσ=dz2, dπ=dxz, dδ=dx2-y2）。
"""
import math, functools, torch
import sympy as sp

_xi, _eta = sp.symbols("xi eta")

def _theta_poly(l, m, c):
    """実球面調和関数の θ 部分のうち cosθ に依存する多項式（sinθ^m は別に扱う）と規格化定数（φ 部分込み）"""
    if (l, m) == (0, 0): return sp.Integer(1), 1 / math.sqrt(4 * math.pi)
    if (l, m) == (1, 0): return c, math.sqrt(3 / (4 * math.pi))
    if (l, m) == (1, 1): return sp.Integer(1), math.sqrt(3 / (4 * math.pi))
    if (l, m) == (2, 0): return 3 * c**2 - 1, math.sqrt(5 / (16 * math.pi))
    if (l, m) == (2, 1): return c, math.sqrt(15 / (4 * math.pi))
    if (l, m) == (2, 2): return sp.Integer(1), math.sqrt(15 / (16 * math.pi))
    raise ValueError((l, m))

@functools.lru_cache(maxsize=None)
def _coeffs(na, la, nb, lb, m):
    """∫∫ ξ^i η^j の係数 c_ij（ξ, η の多項式部分）と角度の規格化・φ 積分の積"""
    u, v = _xi + _eta, _xi - _eta
    ca, cb = (1 + _xi * _eta) / u, (_xi * _eta - 1) / v
    pa, Na = _theta_poly(la, m, ca); pb, Nb = _theta_poly(lb, m, cb)
    sinm = ((_xi**2 - 1) * (1 - _eta**2))**m / (u * v)**m     # sinθa^m sinθb^m
    expr = u**(na - 1) * v**(nb - 1) * pa * pb * sinm * (_xi**2 - _eta**2)
    poly = sp.Poly(sp.expand(sp.cancel(expr)), _xi, _eta)
    terms = [(int(i), int(j), float(c)) for (i, j), c in poly.terms()]
    phi = 2 * math.pi if m == 0 else math.pi
    return terms, Na * Nb * phi

def _A(kmax, p):
    """A_k(p), k = 0..kmax（p > 0）。漸化式 A_k = (e^{-p} + k A_{k-1}) / p"""
    e = torch.exp(-p); out = [e / p]
    for k in range(1, kmax + 1): out.append((e + k * out[-1]) / p)
    return out

_BC = {}
def _bcoef(kmax, nser, dtype, device):
    key = (kmax, nser, dtype, device)
    if key not in _BC:
        c = torch.zeros(nser, kmax + 1, dtype=dtype)
        for k in range(kmax + 1):
            for mm in range(nser):
                if (k + mm) % 2 == 0: c[mm, k] = 2.0 / ((k + mm + 1) * math.factorial(mm))
        _BC[key] = c.to(device)
    return _BC[key]

def _B(kmax, t, nser=60, thr=8.0):
    """B_k(t), k = 0..kmax。|t| < thr では級数 Σ_m (-t)^m/m! · (1+(-1)^{k+m})/(k+m+1)、それ以外は漸化式"""
    small = t.abs() < thr
    ts = torch.where(small, t, torch.zeros_like(t))
    pw = torch.stack([(-ts)**mm for mm in range(nser)], -1)          # (..., nser)
    ser = pw @ _bcoef(kmax, nser, t.dtype, t.device)                 # (..., kmax+1)
    tl = torch.where(small, torch.full_like(t, thr), t)
    ep, em = torch.exp(tl), torch.exp(-tl)
    big = [(ep - em) / tl]
    for k in range(1, kmax + 1):
        big.append(((-1)**k * ep - em + k * big[-1]) / tl)
    return [torch.where(small, ser[..., k], big[k]) for k in range(kmax + 1)]

def overlap_local(na, la, nb, lb, m, za, zb, R):
    """<n_a l_a m (ζa) on a | n_b l_b m (ζb) on b>、R は bohr。za, zb, R は同じ形の tensor"""
    terms, ang = _coeffs(na, la, nb, lb, m)
    p = 0.5 * R * (za + zb); t = 0.5 * R * (za - zb)
    imax = max(i for i, _, _ in terms); jmax = max(j for _, j, _ in terms)
    A = _A(imax, p); B = _B(jmax, t)
    s = torch.zeros_like(R)
    for i, j, c in terms: s = s + c * A[i] * B[j]
    norm = (2 * za)**(na + 0.5) * (2 * zb)**(nb + 0.5) / math.sqrt(math.factorial(2 * na) * math.factorial(2 * nb))
    return ang * norm * (0.5 * R)**(na + nb + 1) * s


# ---------------------------------------------------------------------------
# 分子座標系の重なり行列（回転も自前で行う）
# PYSEQM の diat_overlapD の回転は d–d ブロックで向きによって誤る（Cl2 の H core のノルムが
# 結合の向きで変わる）ので、局所系の積分 S_loc を実球面調和関数の回転で分子系へ移す。
# 軌道の並びは PYSEQM/MOPAC と同じ: s, px, py, pz, dx2-y2, dxz, dz2, dyz, dxy
_S3 = math.sqrt(3.0)
_Q = torch.tensor([[[_S3 / 2, 0, 0], [0, -_S3 / 2, 0], [0, 0, 0]],      # x2-y2
                   [[0, 0, _S3 / 2], [0, 0, 0], [_S3 / 2, 0, 0]],      # xz
                   [[-0.5, 0, 0], [0, -0.5, 0], [0, 0, 1.0]],          # z2
                   [[0, 0, 0], [0, 0, _S3 / 2], [0, _S3 / 2, 0]],      # yz
                   [[0, _S3 / 2, 0], [_S3 / 2, 0, 0], [0, 0, 0]]])     # xy
# 局所系の軌道の種類: (l, m, 型) 型 0 = cos 型（x 側）, 1 = sin 型（y 側）
_LOC = [(0, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 0), (2, 2, 0), (2, 1, 0), (2, 0, 0), (2, 1, 1), (2, 2, 1)]

def local_frames(xij):
    """xij（i→j の単位ベクトル）から局所系の軸 R（行 = x', y', z'）を作る。z' = xij。
    x' の選び方は結果に影響しない（S_loc が z' 周りに対称なため）"""
    z = xij / xij.norm(dim=-1, keepdim=True)
    ref = torch.where((z[..., 2:3].abs() < 0.9), torch.tensor([0.0, 0.0, 1.0], dtype=z.dtype, device=z.device),
                      torch.tensor([1.0, 0.0, 0.0], dtype=z.dtype, device=z.device))
    xp = ref - (ref * z).sum(-1, keepdim=True) * z; xp = xp / xp.norm(dim=-1, keepdim=True)
    yp = torch.cross(z, xp, dim=-1)
    return torch.stack([xp, yp, z], -2)

def orbital_transform(R):
    """C[a, m]: 分子系の軌道 a を局所系の軌道 m で展開した係数（9×9、直交行列）"""
    n = R.shape[0]; Cm = R.new_zeros(n, 9, 9); Cm[:, 0, 0] = 1.0
    # p: 分子系 p_a = Σ_m R[m, a] p'_m（局所の並び x', y', z' → 位置 1, 2, 3）
    Cm[:, 1:4, 1:4] = R.transpose(-1, -2)
    # d: 局所 d'_m(r) = r^T (R^T Q_m R) r = Σ_a c_am d_a  →  d_a = Σ_m c_am d'_m（直交）
    Q = _Q.to(R.dtype).to(R.device)
    Rt = R.transpose(-1, -2)
    Ql = torch.einsum("nij,mjk,nlk->nmil", Rt, Q, Rt)        # (n, m, 3, 3) = R^T Q_m R
    c = torch.einsum("aij,nmij->nam", Q, Ql) / 1.5            # (n, a, m)
    # 局所系の d の並び（x2-y2, xz, z2, yz, xy）は _Q と同じ
    Cm[:, 4:9, 4:9] = c
    return Cm

def overlap_molecular(ni, nj, xij, rij, zeta_a, zeta_b, qn_int, qnD_int):
    """PYSEQM の diatom_overlap_matrixD と同じ引数・出力形 (npairs, 9, 9)。di[k, μ, ν] = <μ(i)|ν(j)>"""
    npairs = rij.shape[0]
    Sloc = rij.new_zeros(npairs, 9, 9)
    na_sp, nb_sp = qn_int[ni], qn_int[nj]; na_d, nb_d = qnD_int[ni], qnD_int[nj]
    for ia, (la, ma, ta) in enumerate(_LOC):
        for ib, (lb, mb, tb) in enumerate(_LOC):
            if ma != mb or ta != tb: continue
            na = na_d if la == 2 else na_sp; nb = nb_d if lb == 2 else nb_sp
            za, zb = zeta_a[..., la], zeta_b[..., lb]
            ok = (za > 0) & (zb > 0) & (na > 0) & (nb > 0)
            if not ok.any(): continue
            keys = torch.stack([na, nb], -1)[ok]
            col = rij.new_zeros(npairs)
            for key in torch.unique(keys, dim=0).tolist():
                sel = ok & (na == key[0]) & (nb == key[1])
                col = col.index_put((sel.nonzero(as_tuple=True)[0],),
                                    overlap_local(key[0], la, key[1], lb, ma, za[sel], zb[sel], rij[sel]))
            Sloc[:, ia, ib] = col
    Cm = orbital_transform(local_frames(xij))
    return Cm @ Sloc @ Cm.transpose(-1, -2)
