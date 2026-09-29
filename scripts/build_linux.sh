#!/usr/bin/env bash
# CalculatoRTX - compilation Linux (CachyOS / Arch / Ubuntu).
# Usage : scripts/build_linux.sh [options CMake supplémentaires]
#   ex. : scripts/build_linux.sh -DCRTX_ENABLE_DLSS=OFF
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build/linux-release}"

# --- CUDA (paquet Arch : /opt/cuda)
if ! command -v nvcc >/dev/null 2>&1; then
    for d in /opt/cuda /usr/local/cuda; do
        if [ -x "$d/bin/nvcc" ]; then export PATH="$d/bin:$PATH"; break; fi
    done
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc introuvable : installez le CUDA Toolkit (paquet 'cuda' sur CachyOS/Arch)." >&2
    exit 1
fi
NVCC="$(command -v nvcc)"
CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$NVCC")")")"
echo "CUDA : $("$NVCC" --version | tail -n 1)"

# --- compilateur hôte de nvcc : sur Arch, GCC le plus récent peut être trop neuf pour nvcc ;
#     le paquet 'cuda' fournit la version compatible (NVCC_CCBIN ou /opt/cuda/bin/g++).
EXTRA=()
if [ -z "${CUDAHOSTCXX:-}" ]; then
    if [ -n "${NVCC_CCBIN:-}" ]; then
        CUDAHOSTCXX="$NVCC_CCBIN"
    elif [ -x "$CUDA_ROOT/bin/g++" ]; then
        CUDAHOSTCXX="$CUDA_ROOT/bin/g++"
    fi
fi
if [ -n "${CUDAHOSTCXX:-}" ]; then
    echo "Compilateur hôte CUDA : $CUDAHOSTCXX"
    EXTRA+=("-DCMAKE_CUDA_HOST_COMPILER=$CUDAHOSTCXX")
fi

GEN=()
if command -v ninja >/dev/null 2>&1; then GEN=(-G Ninja); fi

cmake -S "$ROOT" -B "$BUILD" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$NVCC" "${EXTRA[@]}" "$@"
cmake --build "$BUILD" --parallel
ctest --test-dir "$BUILD" --output-on-failure

echo
echo "Compilation terminée : $BUILD/bin/CalculatoRTX"
