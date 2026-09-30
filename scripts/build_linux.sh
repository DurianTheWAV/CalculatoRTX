#!/usr/bin/env bash
# CalculatoRTX - Linux build (CachyOS / Arch / Ubuntu), NVIDIA or AMD/Intel.
# Usage: scripts/build_linux.sh [extra CMake options]
#   e.g.: scripts/build_linux.sh -DCRTX_ENABLE_CUDA=OFF     (Vulkan backend only)
#         scripts/build_linux.sh -DCRTX_ENABLE_DLSS=OFF
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build/linux-release}"
EXTRA=()

# --- GLSL compiler (Vulkan shaders)
if ! command -v glslangValidator >/dev/null 2>&1; then
    echo "glslangValidator not found: install 'glslang' (CachyOS/Arch) or 'glslang-tools' (Ubuntu)." >&2
    exit 1
fi

# --- CUDA (optional: NVIDIA backend). Arch package: /opt/cuda
if ! command -v nvcc >/dev/null 2>&1; then
    for d in /opt/cuda /usr/local/cuda; do
        if [ -x "$d/bin/nvcc" ]; then export PATH="$d/bin:$PATH"; break; fi
    done
fi
if command -v nvcc >/dev/null 2>&1 && [[ " $* " != *"CRTX_ENABLE_CUDA=OFF"* ]]; then
    NVCC="$(command -v nvcc)"
    CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$NVCC")")")"
    echo "CUDA: $("$NVCC" --version | tail -n 1)  -> NVIDIA backend (OptiX + DLSS) + Vulkan backend"
    EXTRA+=("-DCMAKE_CUDA_COMPILER=$NVCC")
    # nvcc host compiler: on Arch the newest GCC may be too new for nvcc; the 'cuda'
    # package ships the compatible one (NVCC_CCBIN or /opt/cuda/bin/g++).
    if [ -z "${CUDAHOSTCXX:-}" ]; then
        if [ -n "${NVCC_CCBIN:-}" ]; then
            CUDAHOSTCXX="$NVCC_CCBIN"
        elif [ -x "$CUDA_ROOT/bin/g++" ]; then
            CUDAHOSTCXX="$CUDA_ROOT/bin/g++"
        fi
    fi
    if [ -n "${CUDAHOSTCXX:-}" ]; then
        echo "CUDA host compiler: $CUDAHOSTCXX"
        EXTRA+=("-DCMAKE_CUDA_HOST_COMPILER=$CUDAHOSTCXX")
    fi
else
    echo "No CUDA toolkit: building the Vulkan backend only (AMD / Intel / NVIDIA, ray query + FSR 1)"
    EXTRA+=("-DCRTX_ENABLE_CUDA=OFF")
fi

GEN=()
if command -v ninja >/dev/null 2>&1; then GEN=(-G Ninja); fi

cmake -S "$ROOT" -B "$BUILD" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release "${EXTRA[@]}" "$@"
cmake --build "$BUILD" --parallel
ctest --test-dir "$BUILD" --output-on-failure

echo
echo "Build finished: $BUILD/bin/CalculatoRTX"
