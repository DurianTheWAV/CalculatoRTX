#!/usr/bin/env bash
# CalculatoRTX - installation des dépendances sur CachyOS / Arch Linux.
set -euo pipefail

sudo pacman -S --needed \
    base-devel git cmake ninja \
    cuda \
    vulkan-headers vulkan-icd-loader vulkan-tools \
    glfw

echo
echo "Dépendances installées."
echo " - Pilote NVIDIA : 'nvidia-smi' doit fonctionner (paquets nvidia-utils + module nvidia/nvidia-open)."
echo "   Il fournit aussi OptiX (libnvoptix.so) et le cœur NGX du DLSS (libnvidia-ngx.so)."
echo " - Rechargez votre shell (ou 'source /etc/profile.d/cuda.sh') pour avoir nvcc dans le PATH."
echo " - Les en-têtes OptiX et le SDK DLSS sont téléchargés automatiquement par CMake."
