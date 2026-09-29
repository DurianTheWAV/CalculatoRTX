// CalculatoRTX - motif de la grille perforée (trous hexagonaux).
//
// La même fonction de distance signée sert (1) sur l'hôte à "cuire" les Opacity
// Micromaps (OMM, accélérées matériellement par les RT cores Ada) et (2) sur le GPU dans
// le programme any-hit, qui n'est invoqué que pour les micro-triangles "indéterminés".
#pragma once

#include "../common/VecMath.h"

namespace crtx {
namespace grille {

constexpr float kWidth = 3.1f;    // dimensions de la plaque (unités monde)
constexpr float kHeight = 3.3f;
constexpr float kPitch = 0.235f;  // distance entre trous
constexpr float kRadius = 0.088f; // rayon des trous
constexpr float kBorder = 0.16f;  // bord plein
constexpr int kSubdivisionLevel = 5;  // 4^5 = 1024 micro-triangles par triangle

// Distance signée (unités monde) au bord du trou le plus proche : > 0 => matière.
// Fonction 1-lipschitzienne (minimum de distances), ce qui permet un classement exact
// des micro-triangles lors de la construction des OMM.
CRTX_HD float sdf(float2 uv)
{
    const float qx = uv.x * kWidth, qy = uv.y * kHeight;
    const float rowH = kPitch * 0.8660254f;
    const int j0 = static_cast<int>(floorf(qy / rowH));
    float best = 1e9f;
    for (int dj = -1; dj <= 1; ++dj) {
        const int j = j0 + dj;
        const float cy = (static_cast<float>(j) + 0.5f) * rowH;
        const float off = (j & 1) ? kPitch * 0.5f : 0.0f;
        const int i0 = static_cast<int>(floorf((qx - off) / kPitch));
        for (int di = -1; di <= 1; ++di) {
            const float cx = (static_cast<float>(i0 + di) + 0.5f) * kPitch + off;
            // trous trop proches du bord : supprimés (ensemble fixe => reste lipschitzien)
            if (cx < kBorder + kRadius || cx > kWidth - kBorder - kRadius) continue;
            if (cy < kBorder + kRadius || cy > kHeight - kBorder - kRadius) continue;
            const float dx = qx - cx, dy = qy - cy;
            const float d = sqrtf(dx * dx + dy * dy) - kRadius;
            best = fminf(best, d);
        }
    }
    return fminf(best, kPitch);
}

}  // namespace grille
}  // namespace crtx
