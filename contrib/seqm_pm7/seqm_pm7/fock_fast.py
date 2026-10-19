"""d 経路（PYSEQM の method="PM6"、9 軌道）の Fock 行列の組み立てを速くする

PYSEQM は d 経路で全原子対を 45×45 の積分ブロックで扱い、J 項を「掛けてから和」、
交換項（K）を大きな gather で組み立てる。ここでは結果を変えずに次のように置き換える。

1. 原子対を「d 軌道をもつ原子を含む対」と「s・p だけの対」に分ける。
   s・p だけの対は w の 10×10 部分（4 軌道）だけで計算する（外側は厳密にゼロ）。
2. J 項は bmm、K 項は SCF の前に一度だけ並べ替えた行列 G[p,(j,i),(ν,σ)] = w[p, ind[i,ν], ind[j,σ]] との bmm にする。
   G は勾配を記録しない SCF 中だけ w に付けて使い回す（最後の Fock では作り直して自動微分を通す）。
3. 1 中心の d 項（W 積分）は、W がゼロでない行（d 軌道をもつ原子の対角ブロック）だけで計算する。

数式は seqm/seqm_functions/fock.py の _two_center（PM6 分岐）と _d_contrib_one_center と同じ。
"""
import torch
import seqm.seqm_functions.fock as FK

_ORIG_TWO = FK._two_center
_ORIG_D1 = FK._d_contrib_one_center
ENABLED = True


def _groups(w):
    """(d を含む対の添字, s・p だけの対の添字)。w ごとに一度だけ求める"""
    c = getattr(w, "_seqm_pm7_groups", None)
    if c is None:
        with torch.no_grad():
            hasd = (w[:, 10:, :].abs().amax(dim=(1, 2)) > 0) | (w[:, :, 10:].abs().amax(dim=(1, 2)) > 0)
        c = (torch.nonzero(hasd).flatten(), torch.nonzero(~hasd).flatten())
        w._seqm_pm7_groups = c
    return c


def _G(wsub, ind, nb):
    """G[p, j*nb+i, ν*nb+σ] = w[p, ind[i,ν], ind[j,σ]]"""
    a = ind.reshape(-1)                                    # (i,ν) → packed
    g = wsub[:, a, :][:, :, a]                             # (p, (i,ν), (j,σ))
    g = g.view(-1, nb, nb, nb, nb)                         # p, i, ν, j, σ
    return g.permute(0, 3, 1, 2, 4).reshape(-1, nb * nb, nb * nb)   # p, (j,i), (ν,σ)


def _cache(w):
    d = getattr(w, "_seqm_pm7_cache", None)
    if d is None:
        d = {}; w._seqm_pm7_cache = d
    return d


def _sub(w, key, sel, npk):
    """w[sel, :npk, :npk]（先に切り出してから対を拾う）。勾配を記録しない SCF 中は使い回す"""
    if torch.is_grad_enabled() and w.requires_grad:
        return w[:, :npk, :npk][sel]
    d = _cache(w)
    if ("ws", key) not in d:
        d[("ws", key)] = w[:, :npk, :npk][sel].contiguous()
    return d[("ws", key)]


def _cachedG(w, key, ws, ind, nb):
    if torch.is_grad_enabled() and w.requires_grad:
        return _G(ws, ind, nb)
    d = _cache(w)
    if ("G", key) not in d:
        d[("G", key)] = _G(ws, ind, nb)
    return d[("G", key)]


def _two_center_group(F, P, Ppk, w, maskd, mask, idxi, idxj, sel, nb, key):
    if sel.numel() == 0:
        return F
    npk = nb * (nb + 1) // 2
    dev, dt = P.device, P.dtype
    i0, i1 = torch.tril_indices(nb, nb, device=dev)
    ind = FK._cached_index(FK.K_ind_9, dev)[:nb, :nb]
    idxA, idxB = idxj[sel], idxi[sel]
    ws = _sub(w, key, sel, npk)
    PA = Ppk[idxA][:, :npk]                                            # 原子ごとに詰めた P（重み 2 つき）
    PB = Ppk[idxB][:, :npk]
    J_A = torch.bmm(PA.unsqueeze(1), ws).squeeze(1)                   # Σ_a PA[a] w[a,b]
    J_B = torch.bmm(ws, PB.unsqueeze(-1)).squeeze(-1)                 # Σ_b w[a,b] PB[b]
    B = ws.shape[0]
    sumA = torch.zeros(B, 9, 9, device=dev, dtype=dt)
    sumB = torch.zeros_like(sumA)
    sumA[:, i1, i0] = J_A
    sumB[:, i1, i0] = J_B
    F.index_add_(0, maskd[idxA], sumB)
    F.index_add_(0, maskd[idxB], sumA)
    Pp = (-0.5 * P[mask[sel]][:, :nb, :nb]).transpose(1, 2)           # PYSEQM と同じ転置
    G = _cachedG(w, key, ws, ind, nb)
    K = torch.bmm(G, Pp.reshape(B, nb * nb, 1)).view(B, nb, nb)        # K[p, j, i]
    Ks = torch.zeros(B, 9, 9, device=dev, dtype=dt)
    Ks[:, :nb, :nb] = K
    F.index_add_(0, mask[sel], Ks)
    return F


def two_center(F, P, w, maskd, mask, idxi, idxj, themethod):
    if not ENABLED or themethod != "PM6":
        return _ORIG_TWO(F, P, w, maskd, mask, idxi, idxj, themethod)
    seld, selsp = _groups(w)
    dev, dt = P.device, P.dtype
    i0, i1 = FK._cached_index(FK.TRIL_IDX_9, dev)
    wt = FK._cached_tensor(FK.WEIGHT_45, dev, dt)
    Ppk = P[maskd][:, i1, i0] * wt                                     # (原子数, 45)。TRIL の先頭 10 が s・p
    F = _two_center_group(F, P, Ppk, w, maskd, mask, idxi, idxj, seld, 9, "d")
    F = _two_center_group(F, P, Ppk, w, maskd, mask, idxi, idxj, selsp, 4, "sp")
    return F


def _wmat(W):
    """Floc[:, col] = Σ_k W[:, w_idxs[k]] · Pnew[:, p_idxs[k]] を Floc = Wm @ Pnew にする行列（W ごとに一度）"""
    c = getattr(W, "_seqm_pm7_wmat", None)
    if c is None:
        with torch.no_grad():
            rows = torch.nonzero(W.abs().sum(1) > 0).flatten()
        cols, ps, ws = [], [], []
        for col, w_idxs, p_idxs in FK.PM6_FLOCAL_MAP:
            for wi, pi in zip(w_idxs, p_idxs):
                cols.append(col); ps.append(pi); ws.append(wi)
        dev = W.device
        cols = torch.tensor(cols, device=dev); ps = torch.tensor(ps, device=dev); ws = torch.tensor(ws, device=dev)
        Wr = W[rows]
        Wm = torch.zeros(rows.numel(), 45 * 45, dtype=W.dtype, device=dev)
        Wm.index_add_(1, cols * 45 + ps, Wr[:, ws])
        c = (rows, Wm.view(-1, 45, 45))
        W._seqm_pm7_wmat = c
    return c


def d_contrib_one_center(F, P, W, maskd):
    if not ENABLED:
        return _ORIG_D1(F, P, W, maskd)
    rows, Wm = _wmat(W)
    if rows.numel() == 0:
        return F
    i0, i1 = FK._cached_index(FK.TRIL_IDX_9, P.device)
    sc = FK._cached_tensor(FK.WEIGHT_45, P.device, P.dtype)
    Pnew = P[rows][:, i0, i1] * sc
    Floc = torch.bmm(Wm, Pnew.unsqueeze(-1)).squeeze(-1)
    add = torch.zeros(rows.numel(), 9, 9, device=P.device, dtype=P.dtype)
    add[:, i1, i0] = Floc
    F.index_add_(0, rows, add)
    return F


def patch():
    FK._two_center = two_center
    FK._d_contrib_one_center = d_contrib_one_center
