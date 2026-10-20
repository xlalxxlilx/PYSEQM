"""単精度の固有値計算 + 占有–非占有ブロックの Ogita–Aishima 補正（FP64、GEMM は尾崎スキーム II）で密度行列を作る"""
import torch, oz2

def density_mixed(F, nocc, moduli=14, mm=None):
    """F: (n,n) float64 CUDA。戻り値 P = 2 X_occ X_occ^T（X は補正後の占有軌道）"""
    if mm is None:
        mm = lambda a, b: oz2.matmul(a, b, moduli=moduli)
    n = F.shape[-1]
    e32, V32 = torch.linalg.eigh(F.float())
    X = V32.double()
    FX = mm(F, X)
    S = mm(X.T.contiguous(), FX)                       # X^T F X
    R = -mm(X.T.contiguous(), X); R.diagonal().add_(1.0)   # I − X^T X
    lam = S.diagonal() / (1.0 - R.diagonal())
    o = slice(0, nocc); v = slice(nocc, n)
    # E[:, occ] の列だけ作る（補正後の占有軌道 X_occ' = X_occ + X E[:, occ]）
    E = R[:, o] / 2                                    # 同じブロック（occ–occ）は正規直交化だけ
    dl = lam[o].unsqueeze(0) - lam[v].unsqueeze(1)     # λ_j(occ) − λ_i(virt)
    E[v, :] = (S[v, o] + lam[o].unsqueeze(0) * R[v, o]) / dl
    Xo = X[:, o] + mm(X, E.contiguous())
    return 2.0 * mm(Xo, Xo.T.contiguous())
