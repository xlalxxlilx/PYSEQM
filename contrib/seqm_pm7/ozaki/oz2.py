"""GEMMul8（尾崎スキーム II）の DGEMM を torch の CUDA テンソルから呼ぶ（ctypes）"""
import ctypes, os, torch
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "liboz2shim.so"))
_lib.oz2_dgemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.c_double, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                           ctypes.c_double, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_lib.oz2_dgemm.restype = ctypes.c_int
_lib.oz2_workspace_bytes.restype = ctypes.c_size_t
BACKENDS = {"int8": 0, "fp8": 1}

def matmul(A, B, moduli=14, backend="int8", fastmode=True, out=None):
    """C = A @ B（2 次元、float64、CUDA）。行優先の C = A B を列優先の C^T = B^T A^T として計算する"""
    assert A.dtype == torch.float64 and B.dtype == torch.float64 and A.is_cuda and A.dim() == 2 and B.dim() == 2
    A = A.contiguous(); B = B.contiguous()
    M, K = A.shape; K2, N = B.shape; assert K == K2
    C = torch.empty(M, N, dtype=torch.float64, device=A.device) if out is None else out
    st = torch.cuda.current_stream(A.device).cuda_stream
    r = _lib.oz2_dgemm(BACKENDS[backend], 0, 0, N, M, K, 1.0, B.data_ptr(), N, A.data_ptr(), K, 0.0, C.data_ptr(), N,
                       int(moduli), int(bool(fastmode)), ctypes.c_void_p(st))
    if r != 0:
        raise RuntimeError(f"oz2_dgemm failed: {r}")
    return C

def bmm(A, B, **kw):
    """バッチ（3 次元）は 1 枚ずつ（GEMMul8 にバッチ版は無い）"""
    return torch.stack([matmul(a, b, **kw) for a, b in zip(A, B)])

def workspace_bytes():
    return _lib.oz2_workspace_bytes()
