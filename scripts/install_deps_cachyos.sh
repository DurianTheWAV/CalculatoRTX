#!/usr/bin/env bash
# CalculatoRTX - install build dependencies on CachyOS / Arch Linux.
#   NVIDIA GPU : CUDA toolkit (NVIDIA backend: OptiX + DLSS + CUDA) + Vulkan
#   AMD GPU    : Mesa RADV Vulkan driver (Vulkan backend: ray query + FSR 1)
#   Intel GPU  : Mesa ANV Vulkan driver (Vulkan backend)
# Usage: scripts/install_deps_cachyos.sh [--cuda | --no-cuda]
set -euo pipefail

vendor_present() {  # $1 = PCI vendor id (0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel)
    grep -qsi "$1" /sys/class/drm/card*/device/vendor 2>/dev/null
}

WANT_CUDA=auto
for a in "$@"; do
    case "$a" in
        --cuda) WANT_CUDA=yes ;;
        --no-cuda) WANT_CUDA=no ;;
    esac
done
if [ "$WANT_CUDA" = auto ]; then
    if vendor_present 0x10de; then WANT_CUDA=yes; else WANT_CUDA=no; fi
fi

PKGS=(base-devel git cmake ninja glfw glslang vulkan-headers vulkan-icd-loader vulkan-tools)
[ "$WANT_CUDA" = yes ] && PKGS+=(cuda)
vendor_present 0x1002 && PKGS+=(vulkan-radeon)
vendor_present 0x8086 && PKGS+=(vulkan-intel)

echo "Packages: ${PKGS[*]}"
sudo pacman -S --needed "${PKGS[@]}"

echo
echo "Dependencies installed."
if [ "$WANT_CUDA" = yes ]; then
    echo " - NVIDIA: 'nvidia-smi' must work (nvidia-utils + nvidia / nvidia-open module)."
    echo "   The driver also provides OptiX (libnvoptix.so) and the DLSS core (libnvidia-ngx.so)."
    echo " - Reload your shell (or 'source /etc/profile.d/cuda.sh') so that nvcc is in PATH."
    echo "   OptiX headers and the DLSS SDK are downloaded automatically by CMake."
else
    echo " - No CUDA: the Vulkan backend (hardware ray tracing through VK_KHR_ray_query + AMD FSR 1)"
    echo "   will be built. Check that 'vulkaninfo | grep VK_KHR_ray_query' lists the extension"
    echo "   (AMD Radeon RX 6000 series or newer, Intel Arc, NVIDIA RTX)."
fi
