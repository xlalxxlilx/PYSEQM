"""大きい系の SCF の固有値計算を、単精度の固有値計算 + 倍精度の補正（GEMM は尾崎スキーム II）に置き換える

L4 などの倍精度が遅い GPU では、軌道数が 1000 を超えると倍精度の固有値計算（cuSOLVER）が SCF の大半を占める。
ここでは密度行列だけを次の手順で作る。
  1. 単精度で固有値計算（F は float32 に丸める）
  2. Ogita–Aishima（2018）の反復改良を、占有–非占有ブロックだけ 1 回（倍精度）。占有軌道内の回転は P を変えないので、
     縮退していても割り算は起きない。誤差は単精度の誤差の 2 乗程度（L4 で密度行列の差 1e-11 台）
  3. 補正の GEMM（5 回）を GEMMul8（尾崎スキーム II、INT8、法の数 14）で行う
GEMMul8 の窓口 liboz2shim.so（ozaki/shim/）が見つからないときは使わない（倍精度の固有値計算のまま）。
場所は環境変数 SEQM_PM7_OZ2_LIB で指定する。
"""
import ctypes, os, torch

MIN_N = 1000          # これより小さい行列は倍精度の固有値計算のまま（単精度の精度と呼び出しの遅延のため）
MODULI = 14
MAX_COUPLING = 1e-4   # 占有–非占有の補正量がこれを超えたら倍精度の固有値計算に戻す
PASSES = 2            # 補正の回数（2 にすると全体の X を更新してから占有–非占有をもう一度補正する）
STATS = {"mixed": 0, "fp64": 0, "fallback": 0}
_lib = None
_tried = False


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    path = os.environ.get("SEQM_PM7_OZ2_LIB", "")
    if not path or not os.path.exists(path):
        return None
    try:
        lib = ctypes.CDLL(path)
        lib.oz2_dgemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                  ctypes.c_double, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_double, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.oz2_dgemm.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _lib = None
    return _lib


def available():
    return _load() is not None


def matmul(A, B, moduli=MODULI):
    """C = A @ B（2 次元、float64、CUDA）を尾崎スキーム II で。行優先の A B を列優先の B^T A^T として呼ぶ"""
    lib = _load()
    A = A.contiguous(); B = B.contiguous()
    M, K = A.shape; N = B.shape[1]
    C = torch.empty(M, N, dtype=torch.float64, device=A.device)
    st = torch.cuda.current_stream(A.device).cuda_stream
    r = lib.oz2_dgemm(0, 0, 0, N, M, K, 1.0, B.data_ptr(), N, A.data_ptr(), K, 0.0, C.data_ptr(), N,
                      int(moduli), 1, ctypes.c_void_p(st))
    if r != 0:
        raise RuntimeError(f"oz2_dgemm failed: {r}")
    return C


def _density_fp64(F, nocc):
    e, v = torch.linalg.eigh(F, UPLO="U")
    vo = v[:, :nocc]
    return 2.0 * vo @ vo.T


def density_one(F, nocc):
    """1 枚の Fock から閉殻の密度行列。戻り値 (P, 方式)"""
    n = F.shape[-1]
    if n < MIN_N or not F.is_cuda or not available():
        STATS["fp64"] += 1
        return _density_fp64(F, nocc), "fp64"
    mm = matmul
    e32, V32 = torch.linalg.eigh(F.float(), UPLO="U")
    X = V32.double()
    for _ in range(PASSES - 1):                                     # 全体の X を 1 回更新（占有–非占有は OA、同じブロック内は正規化だけ）
        Xt = X.T.contiguous()
        S = mm(Xt, mm(F, X)); R = -mm(Xt, X); R.diagonal().add_(1.0)
        lam = S.diagonal() / (1.0 - R.diagonal())
        E = R / 2
        o, v = slice(0, nocc), slice(nocc, n)
        dl = lam.unsqueeze(0) - lam.unsqueeze(1)
        E[v, o] = (S[v, o] + lam[o].unsqueeze(0) * R[v, o]) / dl[v, o]
        E[o, v] = (S[o, v] + lam[v].unsqueeze(0) * R[o, v]) / dl[o, v]
        X = X + mm(X, E.contiguous())
    Xt = X.T.contiguous()
    S = mm(Xt, mm(F, X))
    R = -mm(Xt, X); R.diagonal().add_(1.0)
    lam = S.diagonal() / (1.0 - R.diagonal())
    o, v = slice(0, nocc), slice(nocc, n)
    E = R[:, o] / 2
    dl = lam[o].unsqueeze(0) - lam[v].unsqueeze(1)
    E[v, :] = (S[v, o] + lam[o].unsqueeze(0) * R[v, o]) / dl
    if not bool(torch.isfinite(E).all()) or float(E[v, :].abs().max()) > MAX_COUPLING:
        STATS["fallback"] += 1
        return _density_fp64(F, nocc), "fallback"
    Xo = X[:, o] + mm(X, E.contiguous())
    STATS["mixed"] += 1
    return 2.0 * mm(Xo, Xo.T.contiguous()), "mixed"


def density(Fc, nocc):
    """(nmol, n, n) → (nmol, n, n)"""
    return torch.stack([density_one(F, nocc)[0] for F in Fc])
