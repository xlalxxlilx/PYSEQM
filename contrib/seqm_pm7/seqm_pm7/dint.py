"""d 軌道を含む二中心二電子積分（MOPAC mndod.F90 の rijkl / charg の移植、微分可能）

PYSEQM の two_elec_two_center_int_local_frame_d_orbitals.py は成分ごとの式を手で書き下しており、
対称性の低い分子で MOPAC と合わない成分がある（回転に依らない誤差。例: 歪んだ SiF4、M(OH)n）。
ここでは MOPAC と同じく、電荷分布 (ij) を多重極（l = 0, 1, 2、電荷間隔 ddp と Klopman–Ohno 項 po）
で表し、ch 表の係数で組み合わせて局所系の積分を作る。局所系は原子 A を原点、B を +z に置き、
両原子で同じ向きの軸を使う（sto_overlap と同じ）。分子系へは sto_overlap.orbital_transform で回す。
sp 同士の積分（MOPAC では古典的な sp の式で計算）は PYSEQM の値をそのまま使う。
"""
import math, functools, torch
from .sto_overlap import local_frames, orbital_transform

# MOPAC の局所系の軌道の並び: 1 s, 2 pz, 3 px, 4 py, 5 dz2, 6 dxz, 7 dyz, 8 dx2-y2, 9 dxy
_LORB = [0, 1, 1, 1, 2, 2, 2, 2, 2]
# MOPAC の並び → sto_overlap の局所系の並び（s, px, py, pz, x2-y2, xz, z2, yz, xy）
_M2S = [0, 3, 1, 2, 6, 5, 7, 4, 8]

def _indexd(i, j):             # 1 始まり、i >= j
    if i < j: i, j = j, i
    return -(j * (j - 1)) // 2 + i + 9 * (j - 1)

_CH = {}
for (ij, l, m), v in {(1,0,0):1.0,(2,1,0):1.0,(3,1,1):1.0,(4,1,-1):1.0,(5,2,0):1.15470054,(6,2,1):1.0,(7,2,-1):1.0,(8,2,2):1.0,(9,2,-2):1.0,
        (10,0,0):1.0,(10,2,0):1.33333333,(11,2,1):1.0,(12,2,-1):1.0,(13,1,0):1.15470054,(14,1,1):1.0,(15,1,-1):1.0,
        (18,0,0):1.0,(18,2,0):-.66666667,(18,2,2):1.0,(19,2,-2):1.0,(20,1,1):-.57735027,(21,1,0):1.0,(23,1,1):1.0,(24,1,-1):1.0,
        (25,0,0):1.0,(25,2,0):-.66666667,(25,2,2):-1.0,(26,1,-1):-.57735027,(28,1,0):1.0,(29,1,-1):-1.0,(30,1,1):1.0,
        (31,0,0):1.0,(31,2,0):1.33333333,(32,2,1):.57735027,(33,2,-1):.57735027,(34,2,2):-1.15470054,(35,2,-2):-1.15470054,
        (36,0,0):1.0,(36,2,0):.66666667,(36,2,2):1.0,(37,2,-2):1.0,(38,2,1):1.0,(39,2,-1):1.0,
        (40,0,0):1.0,(40,2,0):.66666667,(40,2,2):-1.0,(41,2,-1):-1.0,(42,2,1):1.0,(43,0,0):1.0,(43,2,0):-1.33333333,
        (45,0,0):1.0,(45,2,0):-1.33333333}.items():
    _CH[(ij, l, m)] = v

def _lij(li, lj):              # MOPAC indx(li+1, lj+1): ss 1, sp 2, pp 3, sd 4, pd 5, dd 6
    a, b = max(li, lj) + 1, min(li, lj) + 1
    return (a * (a - 1)) // 2 + b

def _charg(r, l1, l2, m, da, db, add):
    s = lambda x: torch.rsqrt(x)
    if l1 == 0 and l2 == 0: return s(r**2 + add)
    if l1 == 1 and l2 == 0: return (-s((r + da)**2 + add) + s((r - da)**2 + add)) / 2
    if l1 == 0 and l2 == 1: return (s((r + db)**2 + add) - s((r - db)**2 + add)) / 2
    if l1 == 1 and l2 == 1 and m == 0:
        return (s((r + da - db)**2 + add) + s((r - da + db)**2 + add) - s((r - da - db)**2 + add) - s((r + da + db)**2 + add)) / 4
    if l1 == 1 and l2 == 1 and m == 1:
        return (2 * s(r**2 + (da - db)**2 + add) - 2 * s(r**2 + (da + db)**2 + add)) * 0.25
    if l1 == 0 and l2 == 2:
        return (s((r - db)**2 + add) - 2 * s(r**2 + db**2 + add) + s((r + db)**2 + add)) / 4
    if l1 == 2 and l2 == 0:
        return (s((r - da)**2 + add) - 2 * s(r**2 + da**2 + add) + s((r + da)**2 + add)) / 4
    if l1 == 1 and l2 == 2 and m == 0:
        return (s((r - da - db)**2 + add) - 2 * s((r - da)**2 + db**2 + add) + s((r + db - da)**2 + add)
                - s((r - db + da)**2 + add) + 2 * s((r + da)**2 + db**2 + add) - s((r + da + db)**2 + add)) / 8
    if l1 == 2 and l2 == 1 and m == 0:
        return (-s((r - da - db)**2 + add) + 2 * s((r - db)**2 + da**2 + add) - s((r + da - db)**2 + add)
                + s((r - da + db)**2 + add) - 2 * s((r + db)**2 + da**2 + add) + s((r + da + db)**2 + add)) / 8
    if l1 == 2 and l2 == 2 and m == 0:
        zzzz = (s((r - da - db)**2 + add) + s((r + da + db)**2 + add) + s((r - da + db)**2 + add) + s((r + da - db)**2 + add)
                - 2 * s((r - da)**2 + db**2 + add) - 2 * s((r - db)**2 + da**2 + add) - 2 * s((r + da)**2 + db**2 + add)
                - 2 * s((r + db)**2 + da**2 + add) + 2 * s(r**2 + (da - db)**2 + add) + 2 * s(r**2 + (da + db)**2 + add))
        xyxy = 4 * s(r**2 + (da - db)**2 + add) + 4 * s(r**2 + (da + db)**2 + add) - 8 * s(r**2 + da**2 + db**2 + add)
        return zzzz / 16 - xyxy / 64
    if l1 == 1 and l2 == 2 and m == 1:
        ab = db / math.sqrt(2.0)
        return (-2 * s((r - ab)**2 + (da - ab)**2 + add) + 2 * s((r + ab)**2 + (da - ab)**2 + add)
                + 2 * s((r - ab)**2 + (da + ab)**2 + add) - 2 * s((r + ab)**2 + (da + ab)**2 + add)) / 8
    if l1 == 2 and l2 == 1 and m == 1:
        aa = da / math.sqrt(2.0)
        return (-2 * s((r + aa)**2 + (aa - db)**2 + add) + 2 * s((r - aa)**2 + (aa - db)**2 + add)
                + 2 * s((r + aa)**2 + (aa + db)**2 + add) - 2 * s((r - aa)**2 + (aa + db)**2 + add)) / 8
    if l1 == 2 and l2 == 2 and m == 1:
        aa = da / math.sqrt(2.0); ab = db / math.sqrt(2.0)
        return (2 * s((r + aa - ab)**2 + (aa - ab)**2 + add) - 2 * s((r + aa + ab)**2 + (aa - ab)**2 + add)
                - 2 * s((r - aa - ab)**2 + (aa - ab)**2 + add) + 2 * s((r - aa + ab)**2 + (aa - ab)**2 + add)
                - 2 * s((r + aa - ab)**2 + (aa + ab)**2 + add) + 2 * s((r + aa + ab)**2 + (aa + ab)**2 + add)
                + 2 * s((r - aa - ab)**2 + (aa + ab)**2 + add) - 2 * s((r - aa + ab)**2 + (aa + ab)**2 + add)) / 16
    if l1 == 2 and l2 == 2 and m == 2:
        return (4 * s(r**2 + (da - db)**2 + add) + 4 * s(r**2 + (da + db)**2 + add) - 8 * s(r**2 + da**2 + db**2 + add)) / 16
    return torch.zeros_like(r)

@functools.lru_cache(maxsize=None)
def _terms(norb_a, norb_b):
    """局所系の (ij|kl) を charg の線形結合で表す表。
    戻り値: dict[(l1, l2, |m|, lij, lkl)] -> list of (ia, ja, kb, lb, coef)（0 始まり、MOPAC の局所軌道番号）"""
    out = {}
    for i in range(1, norb_a + 1):
        for j in range(1, i + 1):
            li, lj = _LORB[i - 1], _LORB[j - 1]; ij = _indexd(i, j); lij = _lij(li, lj)
            for k in range(1, norb_b + 1):
                for l in range(1, k + 1):
                    lk, ll = _LORB[k - 1], _LORB[l - 1]
                    if max(li, lj, lk, ll) < 2: continue          # sp 同士は PYSEQM の値を使う
                    kl = _indexd(k, l); lkl = _lij(lk, ll)
                    for l1 in range(min(abs(li - lj), 2), min(li + lj, 2) + 1):
                        for l2 in range(min(abs(lk - ll), 2), min(lk + ll, 2) + 1):
                            for m in range(-min(l1, l2), min(l1, l2) + 1):
                                c = _CH.get((ij, l1, m), 0.0) * _CH.get((kl, l2, m), 0.0)
                                if c == 0.0: continue
                                out.setdefault((l1, l2, abs(m), lij, lkl), []).append((i - 1, j - 1, k - 1, l - 1, c))
    return out

def _pd(P, l, lij):
    """多重極 l、分布の種類 lij に対する (電荷間隔 D, 加算項 rho)。P は原子ごとの dict（bohr）"""
    if l == 0:
        return None, {1: P["po1"], 3: P["po7"], 6: P["po8"]}[lij]
    return P["ddp"][lij], P["po"][lij]

def local_d_integrals(r, PA, PB, norb_a, norb_b, ev):
    """局所系の積分 W[n, a, b, c, d]（a,b は A の、c,d は B の MOPAC 局所軌道、d を含む成分のみ、eV）"""
    n = r.shape[0]
    W = r.new_zeros(n, norb_a, norb_a, norb_b, norb_b)
    for (l1, l2, m, lij, lkl), lst in _terms(norb_a, norb_b).items():
        da, pa = _pd(PA, l1, lij); db, pb = _pd(PB, l2, lkl)
        zero = torch.zeros_like(r)
        val = _charg(r, l1, l2, m, da if da is not None else zero, db if db is not None else zero, (pa + pb)**2) * ev
        idx = torch.tensor([[a, b, c, d] for a, b, c, d, _ in lst], device=r.device)
        coef = torch.tensor([cc for *_, cc in lst], dtype=r.dtype, device=r.device)
        contrib = val[:, None] * coef[None, :]                                  # (n, nterm)
        for perm in ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)):   # (ij) と (kl) の対称性
            ii = idx[:, list(perm)]
            mask = torch.ones(len(lst), dtype=torch.bool, device=r.device)
            if perm[0] == 1: mask &= idx[:, 0] != idx[:, 1]
            if perm[2] == 3: mask &= idx[:, 2] != idx[:, 3]
            if mask.any():
                sel = ii[mask]; nt = sel.shape[0]
                bidx = torch.arange(n, device=r.device)[:, None].expand(n, nt)
                W = W.index_put((bidx, sel[:, 0].expand(n, nt), sel[:, 1].expand(n, nt), sel[:, 2].expand(n, nt),
                                 sel[:, 3].expand(n, nt)), contrib[:, mask], accumulate=True)
    return W

def molecular_d_integrals(xij, rij, PA, PB, norb_a, norb_b, ev):
    """分子系の (μν|λσ)（d を含む成分）を 9×9×9×9 で返す（sto_overlap の軌道の並び）"""
    Wm = local_d_integrals(rij, PA, PB, norb_a, norb_b, ev)
    n = rij.shape[0]
    Wl = rij.new_zeros(n, 9, 9, 9, 9)
    ia = torch.tensor(_M2S[:norb_a]); ib = torch.tensor(_M2S[:norb_b])
    Wl[:, ia[:, None, None, None], ia[None, :, None, None], ib[None, None, :, None], ib[None, None, None, :]] = Wm
    Cm = orbital_transform(local_frames(xij))
    # 添字を 1 つずつ変換する（9^5 × 4 回）
    Wl = torch.einsum("nap,npqrs->naqrs", Cm, Wl)
    Wl = torch.einsum("nbq,naqrs->nabrs", Cm, Wl)
    Wl = torch.einsum("ncr,nabrs->nabcs", Cm, Wl)
    return torch.einsum("nds,nabcs->nabcd", Cm, Wl)

TRIL = [(a, b) for a in range(9) for b in range(a + 1)]      # PYSEQM の対の並び（行優先の下三角）

def _isY(z):
    """PYSEQM の PM6 で d 軌道をもつ元素（two_elec_two_center_int の YH/YX/YY の判定と同じ）"""
    return (((z > 12) & (z < 18)) | ((z > 20) & (z < 30)) | ((z > 32) & (z < 36)) | ((z > 38) & (z < 48))
            | ((z > 50) & (z < 54)) | ((z > 70) & (z < 80)) | (z == 57))

def replace_d_block(w, ni, nj, xij, rij, PA, PB, ev):
    """PYSEQM の w（npairs, 45, 45、行 = j 側の対、列 = i 側の対）のうち d を含む成分を置き換える"""
    yi, yj = _isY(ni), _isY(nj)
    sel = yi | yj
    if not sel.any(): return w
    w = w.clone()
    norb = lambda z, y: torch.where(y, 9, torch.where(z > 1, 4, 1))
    noa, nob = norb(ni, yi), norb(nj, yj)
    for ka, kb in {(int(a), int(b)) for a, b in zip(noa[sel].tolist(), nob[sel].tolist())}:
        g = (sel & (noa == ka) & (nob == kb)).nonzero(as_tuple=True)[0]
        sub = lambda P: {"ddp": {k: v[g] for k, v in P["ddp"].items()}, "po": {k: v[g] for k, v in P["po"].items()},
                         "po1": P["po1"][g], "po7": P["po7"][g], "po8": P["po8"][g]}
        Wm = molecular_d_integrals(xij[g], rij[g], sub(PA), sub(PB), ka, kb, ev)
        I_list, J_list, a_, b_, c_, d_ = [], [], [], [], [], []
        for I, (a, b) in enumerate(TRIL):
            if a >= (9 if ka == 9 else ka): continue
            for J, (c, d) in enumerate(TRIL):
                if c >= (9 if kb == 9 else kb): continue
                if max(a, b, c, d) < 4: continue
                I_list.append(I); J_list.append(J); a_.append(a); b_.append(b); c_.append(c); d_.append(d)
        if not I_list: continue
        vals = Wm[:, a_, b_, c_, d_]
        gi = g[:, None].expand(-1, len(I_list))
        w = w.index_put((gi, torch.tensor(J_list, device=w.device).expand_as(gi), torch.tensor(I_list, device=w.device).expand_as(gi)), vals)
    return w


# ---------------------------------------------------------------------------
# PM7 だけの処理（mndod.F90 の reppd2 / rotatd）
def _T(a): return a * (a + 3) // 2          # 対 (a, a) の TRIL での位置

def pm7_adjust(w, e1b, e2a, ni, nj, rij, tore, iod_i, iod_j, feather):
    """w: (n,45,45)（行 = j 側の対、列 = i 側の対）、e2a: i 側の電子 × j の芯、e1b: j 側の電子 × i の芯（n,9,9）。
    1. feathering: d を含む成分を const 倍し、対角型（局所系の (aa|cc)、分子系では単位行列×単位行列）には
       (1 − const)·ev/r を足す。芯引力の d ブロックも同様（点電荷 = −ev/r·tore）。
    2. iod > 0 の原子（d 電子をもつ遷移金属）: d と相手の p・s のクーロン積分、自分の p・d の芯引力の平均を
       (ss|ss)・<s|芯> に揃える定数シフト。"""
    yi, yj = _isY(ni), _isY(nj)
    sel = (yi | yj).nonzero(as_tuple=True)[0]
    if len(sel) == 0: return w, e1b, e2a
    w = w.clone(); e1b = e1b.clone(); e2a = e2a.clone()
    const, point = feather(rij[sel])
    # --- feathering（w の d 成分）
    I_d = torch.zeros(45, 45, dtype=torch.bool, device=w.device)       # d を含む成分
    D_d = torch.zeros(45, 45, dtype=w.dtype, device=w.device)          # そのうち対角型
    for I, (a, b) in enumerate(TRIL):
        for J, (c, d) in enumerate(TRIL):
            if max(a, b, c, d) >= 4:
                I_d[J, I] = True
                if a == b and c == d: D_d[J, I] = 1.0
    ws = w[sel]
    ws = torch.where(I_d, ws * const[:, None, None] + (1 - const)[:, None, None] * point[:, None, None] * D_d, ws)
    # --- feathering（芯引力の d ブロック）
    Ed = torch.zeros(9, 9, dtype=torch.bool, device=w.device); Ed[4:, :] = True; Ed[:, 4:] = True
    Eye = torch.eye(9, dtype=w.dtype, device=w.device)
    pj = -point * tore[nj[sel]]; pi_ = -point * tore[ni[sel]]
    e2s = torch.where(Ed, e2a[sel] * const[:, None, None] + ((1 - const) * pj)[:, None, None] * Eye, e2a[sel])
    e1s = torch.where(Ed, e1b[sel] * const[:, None, None] + ((1 - const) * pi_)[:, None, None] * Eye, e1b[sel])
    # --- iod > 0
    Di = [_T(a) for a in range(4, 9)]; Pi = [_T(a) for a in range(1, 4)]
    oi, oj = iod_i[sel] > 0, iod_j[sel] > 0
    npi = torch.where(yi[sel], 9, torch.where(ni[sel] > 1, 4, 1)); npj = torch.where(yj[sel], 9, torch.where(nj[sel] > 1, 4, 1))
    w11 = ws[:, 0, 0]
    def shift(block_rows, block_cols, mask, nterm):
        blk = ws[:, block_rows][:, :, block_cols]
        s = w11 - blk.sum(dim=(1, 2)) / nterm
        add = torch.zeros_like(ws); add[:, torch.tensor(block_rows)[:, None], torch.tensor(block_cols)[None, :]] = 1.0
        return ws + torch.where(mask, s, torch.zeros_like(s))[:, None, None] * add
    # i 側が遷移金属: (i の d | j の p)、(i の d | j の s)
    ws = shift(Pi, Di, oi & (npj > 1), 15.0)
    ws = shift([0], Di, oi, 5.0)
    # j 側が遷移金属: (i の d | j の d)（両方）、(i の p | j の d)、(i の s | j の d)
    ws = shift(Di, Di, oi & oj, 25.0)
    ws = shift(Di, Pi, oj & (npi > 1), 15.0)
    ws = shift(Di, [0], oj, 5.0)
    def eshift(E, mask_p, mask_d):
        sp_ = E[:, 0, 0] - (E[:, 1, 1] + E[:, 2, 2] + E[:, 3, 3]) / 3
        sd_ = E[:, 0, 0] - sum(E[:, k, k] for k in range(4, 9)) / 5
        Ip = torch.zeros(9, 9, dtype=E.dtype, device=E.device); Ip[1, 1] = Ip[2, 2] = Ip[3, 3] = 1
        Id = torch.zeros(9, 9, dtype=E.dtype, device=E.device)
        for k in range(4, 9): Id[k, k] = 1
        return E + torch.where(mask_p, sp_, 0 * sp_)[:, None, None] * Ip + torch.where(mask_d, sd_, 0 * sd_)[:, None, None] * Id
    e2s = eshift(e2s, oi & (npj > 1), oi)
    e1s = eshift(e1s, oj & (npi > 1), oj)
    w = w.index_put((sel,), ws); e1b = e1b.index_put((sel,), e1s); e2a = e2a.index_put((sel,), e2s)
    return w, e1b, e2a
