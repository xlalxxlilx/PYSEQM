#!/usr/bin/env bash
# GEMMul8（尾崎スキーム II、MIT ライセンス）をビルドし、seqm_pm7 から呼ぶ窓口 liboz2shim.so を作る。
# 必要なもの: CUDA Toolkit 12.9 以上（nvcc、cuBLAS/cuBLASLt の静的ライブラリ）、C++20 対応の g++、git、make。
# 例（conda で CUDA 12.9 を用意する場合）:
#   conda create -n cuda129 -c conda-forge cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-cudart-static=12.9 \
#         libcublas-dev=12.9 libcublas-static=12.9 cuda-nvml-dev=12.9 make
#   conda activate cuda129 && bash build_oz2.sh 89        # 89 = L4 / RTX 40 系（Ada）
# できた liboz2shim.so の場所を環境変数 SEQM_PM7_OZ2_LIB に設定すると、seqm_pm7 の eig="auto" が使う。
# cuBLAS 12.9 は静的リンクし記号を隠すので、torch が読み込む古い cuBLAS とは衝突しない。
set -euo pipefail
ARCH=${1:-89}
CUDA=${CUDA_PATH:-${CONDA_PREFIX:-/usr/local/cuda}}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
[ -d GEMMul8 ] || git clone https://github.com/RIKEN-RCCS/GEMMul8.git
( cd GEMMul8 && git checkout -q 5f3fae8 && make -j"$(nproc)" BACKEND=cuda CUDA_PATH="$CUDA" GPU_ARCH="$ARCH" OPS="gemm" OZ2_BACKENDS="INT8 FP8" CXX=g++ )
nvcc -std=c++20 -O2 -arch=sm_"$ARCH" -shared -Xcompiler "-fPIC -fvisibility=hidden" -IGEMMul8/include oz2_shim.cu \
  -Xlinker --whole-archive GEMMul8/lib/libgemmul8.a -Xlinker --no-whole-archive \
  -L"$CUDA"/lib -L"$CUDA"/lib64 -lcublas_static -lcublasLt_static -lculibos -lcudart_static -lpthread -ldl -lrt \
  -Xlinker --exclude-libs,ALL -o liboz2shim.so
echo "liboz2shim.so を作成: export SEQM_PM7_OZ2_LIB=$HERE/liboz2shim.so"
