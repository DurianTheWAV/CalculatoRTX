// CalculatoRTX - trous hexagonaux de la grille perforée (portage de src/render/Grille.h).
// Backend Vulkan : la géométrie de la grille est déclarée non opaque ; chaque intersection
// candidate est acceptée ou rejetée par cette fonction dans la boucle rayQueryProceedEXT.
#ifndef CRTX_GRILLE_GLSL
#define CRTX_GRILLE_GLSL

const float kGrilleWidth = 3.1;
const float kGrilleHeight = 3.3;
const float kGrillePitch = 0.235;
const float kGrilleRadius = 0.088;
const float kGrilleBorder = 0.16;

// Distance signée (unités monde) au bord du trou le plus proche : > 0 => matière.
float grilleSdf(vec2 uv)
{
    const float qx = uv.x * kGrilleWidth, qy = uv.y * kGrilleHeight;
    const float rowH = kGrillePitch * 0.8660254;
    const int j0 = int(floor(qy / rowH));
    float best = 1e9;
    for (int dj = -1; dj <= 1; ++dj) {
        const int j = j0 + dj;
        const float cy = (float(j) + 0.5) * rowH;
        const float off = (j & 1) != 0 ? kGrillePitch * 0.5 : 0.0;
        const int i0 = int(floor((qx - off) / kGrillePitch));
        for (int di = -1; di <= 1; ++di) {
            const float cx = (float(i0 + di) + 0.5) * kGrillePitch + off;
            if (cx < kGrilleBorder + kGrilleRadius || cx > kGrilleWidth - kGrilleBorder - kGrilleRadius) continue;
            if (cy < kGrilleBorder + kGrilleRadius || cy > kGrilleHeight - kGrilleBorder - kGrilleRadius) continue;
            const float dx = qx - cx, dy = qy - cy;
            best = min(best, sqrt(dx * dx + dy * dy) - kGrilleRadius);
        }
    }
    return min(best, kGrillePitch);
}

#endif
