// CalculatoRTX - génération procédurale de maillages.
#include "Mesh.h"

#include <algorithm>
#include <cmath>

namespace crtx {

namespace {

struct ProfilePoint {
    float d;     // retrait vers l'intérieur depuis le contour extérieur
    float y;     // hauteur depuis la face inférieure
    float nOut;  // composante horizontale (vers l'extérieur) de la normale
    float nUp;   // composante verticale de la normale
};

void pushTriangle(Mesh& m, unsigned a, unsigned b, unsigned c)
{
    const float3 pa = m.positions[a], pb = m.positions[b], pc = m.positions[c];
    const float3 n = cross(pb - pa, pc - pa);
    if (dot(n, n) < 1e-14f) return;  // triangle dégénéré ignoré
    m.indices.push_back(make_uint3(a, b, c));
}

unsigned pushVertex(Mesh& m, float3 p)
{
    m.positions.push_back(p);
    return static_cast<unsigned>(m.positions.size() - 1);
}

}  // namespace

void addRoundedBox(Mesh& mesh, const RoundedBoxDesc& d)
{
    const float hx = d.size.x * 0.5f, hz = d.size.z * 0.5f, h = d.size.y;
    const float dmax = std::min(hx, hz);
    const float rc = std::min(d.cornerRadius, dmax);
    const float rt = std::min(d.topFillet, std::min(h * 0.5f, dmax * 0.9f));
    const float rb = std::min(d.bottomFillet, std::min(h * 0.5f, dmax * 0.9f));
    const int fs = std::max(1, d.filletSegments);
    const int cs = std::max(1, d.cornerSegments);

    std::vector<ProfilePoint> prof;
    // face inférieure
    prof.push_back({dmax, 0.0f, 0.0f, -1.0f});
    if (rb > 0.0f) {
        prof.push_back({rb, 0.0f, 0.0f, -1.0f});
        for (int k = 0; k <= fs; ++k) {
            const float t = (-90.0f + 90.0f * k / fs) * kPi / 180.0f;
            prof.push_back({rb - rb * std::cos(t), rb + rb * std::sin(t), std::cos(t), std::sin(t)});
        }
    } else {
        prof.push_back({0.0f, 0.0f, 0.0f, -1.0f});
        prof.push_back({0.0f, 0.0f, 1.0f, 0.0f});
    }
    // flanc + congé supérieur
    if (rt > 0.0f) {
        for (int k = 0; k <= fs; ++k) {
            const float t = (90.0f * k / fs) * kPi / 180.0f;
            prof.push_back({rt - rt * std::cos(t), h - rt + rt * std::sin(t), std::cos(t), std::sin(t)});
        }
    } else {
        prof.push_back({0.0f, h, 1.0f, 0.0f});
        prof.push_back({0.0f, h, 0.0f, 1.0f});
    }
    // face supérieure (anneaux concentriques, creux parabolique optionnel)
    const int K = std::max(1, d.topRings);
    const float span = std::max(dmax - rt, 1e-4f);
    for (int k = 0; k <= K; ++k) {
        const float t = static_cast<float>(k) / K;
        const float s = 1.0f - t;
        const float dd = rt + span * t;
        const float y = h - d.dish * (1.0f - s * s);
        const float slope = 2.0f * d.dish * s / span;
        const float l = std::sqrt(slope * slope + 1.0f);
        prof.push_back({std::min(dd, dmax), y, -slope / l, 1.0f / l});
    }

    const int ringSize = 4 * (cs + 1);
    const unsigned base = static_cast<unsigned>(mesh.positions.size());
    const bool withUv = true;
    for (const ProfilePoint& p : prof) {
        const float r = std::max(rc - p.d, 0.0f);
        const float inset = std::max(rc, p.d);
        for (int c = 0; c < 4; ++c) {
            const float sx = (c == 0 || c == 3) ? 1.0f : -1.0f;
            const float sz = (c == 0 || c == 1) ? 1.0f : -1.0f;
            for (int j = 0; j <= cs; ++j) {
                const float phi = (c * 90.0f + 90.0f * j / cs) * kPi / 180.0f;
                const float cx = std::cos(phi), cz = std::sin(phi);
                const float x = sx * std::max(hx - inset, 0.0f) + r * cx;
                const float z = sz * std::max(hz - inset, 0.0f) + r * cz;
                mesh.positions.push_back(make_float3(d.center.x + x, d.center.y + p.y, d.center.z + z));
                mesh.normals.push_back(normalize(make_float3(p.nOut * cx, p.nUp, p.nOut * cz)));
                if (withUv) mesh.uvs.push_back(make_float2((x + hx) / (2.0f * hx), (z + hz) / (2.0f * hz)));
            }
        }
    }
    for (size_t r = 0; r + 1 < prof.size(); ++r) {
        const unsigned a0 = base + static_cast<unsigned>(r * ringSize);
        const unsigned b0 = a0 + ringSize;
        for (int i = 0; i < ringSize; ++i) {
            const unsigned i1 = static_cast<unsigned>((i + 1) % ringSize);
            pushTriangle(mesh, a0 + i, a0 + i1, b0 + i1);
            pushTriangle(mesh, a0 + i, b0 + i1, b0 + i);
        }
    }
}

float roundedBoxTopHeight(const RoundedBoxDesc& d, float x, float z)
{
    const float hx = d.size.x * 0.5f, hz = d.size.z * 0.5f, h = d.size.y;
    const float dmax = std::min(hx, hz);
    const float rt = std::min(d.topFillet, std::min(h * 0.5f, dmax * 0.9f));
    const float inset = std::min(hx - std::fabs(x), hz - std::fabs(z));
    if (inset <= rt) return d.center.y + h;
    const float t = std::min((inset - rt) / std::max(dmax - rt, 1e-4f), 1.0f);
    const float s = 1.0f - t;
    return d.center.y + h - d.dish * (1.0f - s * s);
}

void addGridQuad(Mesh& mesh, float3 corner, float3 edgeU, float3 edgeV, int nu, int nv, bool withNormals)
{
    const unsigned base = static_cast<unsigned>(mesh.positions.size());
    const float3 n = normalize(cross(edgeU, edgeV));
    for (int j = 0; j <= nv; ++j) {
        for (int i = 0; i <= nu; ++i) {
            const float u = static_cast<float>(i) / nu, v = static_cast<float>(j) / nv;
            mesh.positions.push_back(corner + edgeU * u + edgeV * v);
            if (withNormals) mesh.normals.push_back(n);
            mesh.uvs.push_back(make_float2(u, v));
        }
    }
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const unsigned a = base + j * (nu + 1) + i;
            const unsigned b = a + 1, c = a + (nu + 1), dd = c + 1;
            mesh.indices.push_back(make_uint3(a, b, dd));
            mesh.indices.push_back(make_uint3(a, dd, c));
        }
    }
}

namespace {

struct StrokeBuilder {
    Mesh& mesh;
    const TextFrame& frame;
    float height;
    const std::function<float(float3)>& baseOffset;

    float3 place(float2 p, float h) const
    {
        const float3 onPlane = frame.origin + frame.right * p.x + frame.up * p.y;
        const float off = baseOffset ? baseOffset(onPlane) : 0.0f;
        return onPlane + frame.normal * (off + h);
    }

    // Prisme à section trapézoïdale entre deux points (extrémités prolongées de extA / extB)
    void segment(float2 a, float2 b, float hw, float bevel, float extA, float extB) const
    {
        float2 dir = b - a;
        const float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        if (len < 1e-6f) return;
        dir = dir * (1.0f / len);
        const float2 n = make_float2(-dir.y, dir.x);
        const float2 A = a - dir * extA, B = b + dir * extB;
        const float bv = std::min(bevel, hw * 0.9f);
        const float2 At = A + dir * std::min(bv, extA), Bt = B - dir * std::min(bv, extB);
        const float th = hw - bv;
        const float hb = -height * 0.35f;  // enfoncé légèrement dans la surface
        unsigned v[8];
        v[0] = pushVertex(mesh, place(A - n * hw, hb));
        v[1] = pushVertex(mesh, place(A + n * hw, hb));
        v[2] = pushVertex(mesh, place(B + n * hw, hb));
        v[3] = pushVertex(mesh, place(B - n * hw, hb));
        v[4] = pushVertex(mesh, place(At - n * th, height));
        v[5] = pushVertex(mesh, place(At + n * th, height));
        v[6] = pushVertex(mesh, place(Bt + n * th, height));
        v[7] = pushVertex(mesh, place(Bt - n * th, height));
        const int quads[5][4] = {{4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
        for (const auto& q : quads) {
            pushTriangle(mesh, v[q[0]], v[q[1]], v[q[2]]);
            pushTriangle(mesh, v[q[0]], v[q[2]], v[q[3]]);
        }
    }

    // Articulation arrondie (prisme octogonal) aux extrémités et aux angles vifs
    void joint(float2 c, float hw, float bevel) const
    {
        constexpr int N = 10;
        const float bv = std::min(bevel, hw * 0.9f);
        const unsigned center = pushVertex(mesh, place(c, height));
        unsigned bot[N], top[N];
        for (int i = 0; i < N; ++i) {
            const float t = kTwoPi * i / N;
            const float2 d = make_float2(std::cos(t), std::sin(t));
            bot[i] = pushVertex(mesh, place(c + d * hw, -height * 0.35f));
            top[i] = pushVertex(mesh, place(c + d * (hw - bv), height));
        }
        for (int i = 0; i < N; ++i) {
            const int j = (i + 1) % N;
            pushTriangle(mesh, center, top[i], top[j]);
            pushTriangle(mesh, bot[i], bot[j], top[j]);
            pushTriangle(mesh, bot[i], top[j], top[i]);
        }
    }
};

float turnAngle(float2 a, float2 b, float2 c)
{
    float2 u = b - a, v = c - b;
    const float lu = std::sqrt(u.x * u.x + u.y * u.y), lv = std::sqrt(v.x * v.x + v.y * v.y);
    if (lu < 1e-6f || lv < 1e-6f) return 0.0f;
    const float cosA = std::max(-1.0f, std::min(1.0f, (u.x * v.x + u.y * v.y) / (lu * lv)));
    return std::acos(cosA);
}

}  // namespace

float addText(Mesh& mesh, const StrokeFont& font, const std::string& text, const TextFrame& frame,
              const TextStyle& style, TextAlign align, const std::function<float(float3)>& baseOffset)
{
    mesh.flat = true;
    const TextLayout lay = font.layout(text);
    const float width = lay.width * style.size;
    float ox = 0.0f;
    if (align == TextAlign::Center) ox = -width * 0.5f;
    else if (align == TextAlign::Right) ox = -width;

    StrokeBuilder sb{mesh, frame, style.height, baseOffset};
    constexpr float kSharp = 0.55f;  // ~31° : au-delà, articulation arrondie
    for (const Polyline2D& pl : lay.strokes) {
        const float hw = 0.5f * style.strokeWidth * style.size * (0.55f + 0.45f * pl.scale);
        const float bevel = hw * 2.0f * style.bevel;
        std::vector<float2> pts;
        pts.reserve(pl.points.size());
        for (const float2& p : pl.points) pts.push_back(make_float2(ox + p.x * style.size, p.y * style.size));
        const size_t n = pts.size();
        const float2 d0 = pts.front() - pts.back();
        const bool closed = n > 3 && (d0.x * d0.x + d0.y * d0.y) < 1e-6f * style.size * style.size;

        auto angleAt = [&](size_t i) -> float {
            if (i == 0 || i == n - 1) {
                if (!closed) return 10.0f;  // extrémité libre : capuchon arrondi
                return turnAngle(pts[n - 2], pts[0], pts[1]);
            }
            return turnAngle(pts[i - 1], pts[i], pts[i + 1]);
        };
        for (size_t i = 0; i + 1 < n; ++i) {
            const float aA = angleAt(i), aB = angleAt(i + 1);
            const float extA = aA < kSharp ? hw * std::tan(aA * 0.5f) + hw * 0.02f : 0.0f;
            const float extB = aB < kSharp ? hw * std::tan(aB * 0.5f) + hw * 0.02f : 0.0f;
            sb.segment(pts[i], pts[i + 1], hw, bevel, extA, extB);
        }
        for (size_t i = 0; i < n; ++i) {
            if (closed && i == n - 1) break;
            if (angleAt(i) >= kSharp) sb.joint(pts[i], hw, bevel);
        }
    }
    return width;
}

}  // namespace crtx
