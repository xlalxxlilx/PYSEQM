// GEMMul8（尾崎スキーム II）の DGEMM を Python（ctypes）から呼ぶための薄い窓口
// 行列は列優先（cuBLAS と同じ）。torch の行優先 C = A @ B は、列優先で C^T = B^T A^T として呼ぶ（Python 側で行う）。
// cuBLAS 12.9 は静的リンクし、記号は外に出さない（torch が読み込む cuBLAS 12.4 と衝突させないため）。
#include "gemmul8.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

static cublasHandle_t g_h = nullptr;
static cublasLtHandle_t g_lt = nullptr;
static void *g_work = nullptr;
static size_t g_lwork = 0;

static int ensure(size_t need) {
    if (need <= g_lwork) return 0;
    if (g_work) cudaFree(g_work);
    g_work = nullptr; g_lwork = 0;
    if (cudaMalloc(&g_work, need) != cudaSuccess) return -2;
    g_lwork = need;
    return 0;
}

extern "C" {

// backend: 0 = INT8（cuBLAS ハンドル）、1 = FP8（cuBLASLt ハンドル）。戻り値: 0 成功、負で失敗
__attribute__((visibility("default")))
int oz2_dgemm(int backend, int transA, int transB, size_t m, size_t n, size_t k,
              double alpha, const double *A, size_t lda, const double *B, size_t ldb,
              double beta, double *C, size_t ldc, int num_moduli, int fastmode, void *stream) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    cublasOperation_t oa = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t ob = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (backend == 0) {
        if (!g_h && cublasCreate(&g_h) != CUBLAS_STATUS_SUCCESS) return -1;
        cublasSetStream(g_h, st);
        auto q = gemmul8::gemm<gemmul8::Backend::INT8>(g_h, oa, ob, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc,
                                                       num_moduli, fastmode != 0, nullptr);
        if (ensure(size_t(q[0]))) return -2;
        gemmul8::gemm<gemmul8::Backend::INT8>(g_h, oa, ob, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc,
                                              num_moduli, fastmode != 0, g_work);
    } else {
        if (!g_lt && cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return -1;
        auto q = gemmul8::gemmLt<gemmul8::Backend::FP8>(g_lt, oa, ob, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc,
                                                        num_moduli, fastmode != 0, nullptr,
                                                        nullptr, nullptr, false, false, false, false, st);
        if (ensure(size_t(q[0]))) return -2;
        gemmul8::gemmLt<gemmul8::Backend::FP8>(g_lt, oa, ob, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc,
                                               num_moduli, fastmode != 0, g_work,
                                               nullptr, nullptr, false, false, false, false, st);
    }
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : -3;
}

__attribute__((visibility("default"))) size_t oz2_workspace_bytes() { return g_lwork; }

__attribute__((visibility("default"))) void oz2_release() {
    if (g_work) cudaFree(g_work);
    g_work = nullptr; g_lwork = 0;
}

}
