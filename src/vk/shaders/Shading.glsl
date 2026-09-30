// CalculatoRTX - shading GPU du backend Vulkan : aléatoire, bruit, motifs procéduraux, BSDF.
// Portage ligne à ligne de src/render/device/Shading.cuh (backend OptiX) : les deux moteurs
// produisent la même image.
#ifndef CRTX_SHADING_GLSL
#define CRTX_SHADING_GLSL

// ============================================================ nombres aléatoires (PCG)
uint pcgHash(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct Rng {
    uint s;
};

float rngNext(inout Rng r)
{
    r.s = pcgHash(r.s);
    return float(r.s >> 8) * (1.0 / 16777216.0);
}

vec2 rngNext2(inout Rng r)
{
    const float a = rngNext(r);
    return vec2(a, rngNext(r));
}

// ============================================================ bruit de valeur 3D
float hash3i(int x, int y, int z)
{
    const uint h = pcgHash(uint(x) * 73856093u ^ pcgHash(uint(y) * 19349663u ^ uint(z) * 83492791u));
    return float(h & 0xFFFFFFu) * (1.0 / 16777215.0);
}

float valueNoise(vec3 p)
{
    const vec3 f = floor(p);
    const int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    vec3 t = p - f;
    t = t * t * (3.0 - 2.0 * t);
    const float c000 = hash3i(ix, iy, iz), c100 = hash3i(ix + 1, iy, iz);
    const float c010 = hash3i(ix, iy + 1, iz), c110 = hash3i(ix + 1, iy + 1, iz);
    const float c001 = hash3i(ix, iy, iz + 1), c101 = hash3i(ix + 1, iy, iz + 1);
    const float c011 = hash3i(ix, iy + 1, iz + 1), c111 = hash3i(ix + 1, iy + 1, iz + 1);
    const float x00 = mix(c000, c100, t.x), x10 = mix(c010, c110, t.x);
    const float x01 = mix(c001, c101, t.x), x11 = mix(c011, c111, t.x);
    return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

float fbm(vec3 p, int octaves)
{
    float a = 0.5, s = 0.0;
    for (int i = 0; i < octaves; ++i) {
        s += a * valueNoise(p);
        p = p * 2.03 + vec3(17.1, 3.7, 9.2);
        a *= 0.5;
    }
    return s;
}

// ============================================================ matériau évalué au point d'impact
struct Surface {
    vec3 base;
    vec3 emission;
    float roughness;
    float metallic;
    float transmission;
    float ior;
    float clearcoat;
    float ccRoughness;
    float specular;
};

// P : position dans le repère de l'objet
Surface evalMaterial(Material m, vec3 P, vec2 uv)
{
    Surface s;
    s.base = m.baseColor;
    s.emission = m.emission;
    s.roughness = m.roughness;
    s.metallic = m.metallic;
    s.transmission = m.transmission;
    s.ior = m.ior;
    s.clearcoat = m.clearcoat;
    s.ccRoughness = m.clearcoatRoughness;
    s.specular = m.specular;

    if (m.pattern == kPatternWood) {
        const vec3 q = P * m.patternScale;
        const float plank = floor(q.z / 3.4);
        const float h = hash3i(int(plank), 7, 11);
        const float w = fbm(vec3(q.x * 0.12 + h * 31.0, q.z * 1.1, h * 5.0), 4);
        const float grain = q.z * 5.0 + w * 3.2 + sin(q.x * 0.21 + h * 6.28) * 0.9;
        float rings = 0.5 + 0.5 * sin(grain * 3.1);
        rings = rings * rings;
        const float fine = valueNoise(vec3(q.x * 0.6, q.z * 40.0, h * 3.0));
        const vec3 dark = vec3(0.055, 0.026, 0.012);
        const vec3 light = vec3(0.24, 0.12, 0.055);
        s.base = mix(dark, light, 0.25 + 0.6 * rings + 0.15 * fine) * (0.8 + 0.35 * h);
        const float seam = fract(q.z / 3.4);
        if (seam < 0.004 || seam > 0.996) s.base = s.base * 0.25;
        s.roughness = m.roughness * (0.8 + 0.5 * fine);
    } else if (m.pattern == kPatternBrushed) {
        const float n = valueNoise(vec3(P.x * 1.5, P.z * 220.0, P.y * 220.0));
        const float n2 = valueNoise(vec3(P.x * 0.3, P.z * 40.0, 3.0));
        s.base = s.base * (0.9 + 0.14 * n + 0.05 * n2);
        s.roughness = m.roughness * (0.75 + 0.5 * n);
    } else if (m.pattern == kPatternSolar) {
        const float fu = fract(uv.x * m.patternScale);
        const float fv = fract(uv.y * 7.0);
        const float n = valueNoise(vec3(uv.x * 60.0, uv.y * 12.0, 1.0));
        s.base = vec3(0.03, 0.018, 0.045) * (0.7 + 0.6 * n);
        if (fu < 0.015 || fu > 0.985) {
            s.base = vec3(0.55, 0.55, 0.58);
            s.metallic = 1.0;
            s.roughness = 0.3;
        } else if (fv < 0.05) {
            s.base = vec3(0.35, 0.35, 0.38);
            s.metallic = 1.0;
            s.roughness = 0.35;
        }
    } else if (m.pattern == kPatternVfd) {
        const float fx = fract(uv.x * 154.0), fy = fract(uv.y * 33.0);
        if (fx < 0.07 || fy < 0.07) s.base = vec3(0.03, 0.07, 0.065);
    } else if (m.pattern == kPatternCarbon) {
        const float sc = m.patternScale;
        const float cx = floor(P.x * sc), cz = floor(P.z * sc);
        const bool warp = (int(cx + cz) & 1) != 0;
        const float t = warp ? fract(P.x * sc) : fract(P.z * sc);
        const float sheen = 0.55 + 0.9 * sin(t * kPi);
        s.base = s.base * sheen;
        s.roughness = m.roughness * (warp ? 0.8 : 1.25);
    } else if (m.pattern == kPatternPcb) {
        const float rx = floor(P.x / 1.3), rz = floor(P.z / 1.3);
        const bool alongX = hash3i(int(rx), int(rz), 3) < 0.5;
        const float along = alongX ? P.x : P.z, across = alongX ? P.z : P.x;
        const float lane = floor(across / 0.1), fl = fract(across / 0.1);
        const float laneSeed = hash3i(int(lane), 5, int(alongX ? rz : rx));
        const float seg = floor(along / 0.7 + laneSeed);
        const float hs = hash3i(int(lane), int(seg), int(rx * 7.0 + rz * 13.0));
        if (hs < 0.55 && abs(fl - 0.5) < 0.2) s.base = s.base * 2.3 + vec3(0.01, 0.03, 0.0);
        const float vx = P.x / 0.45, vz = P.z / 0.45;
        const float hv = hash3i(int(floor(vx)), int(floor(vz)), 9);
        const float dx = fract(vx) - 0.5, dz = fract(vz) - 0.5;
        const float r = sqrt(dx * dx + dz * dz) * 0.45;
        if (hv < 0.18 && r < 0.04) {
            if (r < 0.017) {
                s.base = vec3(0.01);
                s.roughness = 0.8;
            } else {
                s.base = vec3(0.78, 0.78, 0.8);
                s.metallic = 1.0;
                s.roughness = 0.3;
            }
            s.clearcoat = 0.0;
        }
    }
    return s;
}

// ============================================================ BSDF (Lambert + GGX + vernis)
struct Onb {
    vec3 t, b, n;
};

Onb makeOnb(vec3 n)
{
    // Duff et al. 2017 (repère orthonormé sans branche)
    const float sgn = n.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (sgn + n.z);
    const float b = n.x * n.y * a;
    Onb o;
    o.t = vec3(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    o.b = vec3(b, sgn + n.y * n.y * a, -n.y);
    o.n = n;
    return o;
}

vec3 toLocal(Onb o, vec3 v) { return vec3(dot(v, o.t), dot(v, o.b), dot(v, o.n)); }
vec3 toWorld(Onb o, vec3 v) { return o.t * v.x + o.b * v.y + o.n * v.z; }

vec3 schlick3(vec3 f0, float c)
{
    const float t = pow(max(1.0 - c, 0.0), 5.0);
    return f0 + (vec3(1.0) - f0) * t;
}
float schlick1(float f0, float c)
{
    const float t = pow(max(1.0 - c, 0.0), 5.0);
    return f0 + (1.0 - f0) * t;
}

float ggxD(float noh, float a2)
{
    const float d = noh * noh * (a2 - 1.0) + 1.0;
    return a2 / (kPi * d * d + 1e-20);
}

float smithG1(float nov, float a2) { return 2.0 * nov / (nov + sqrt(a2 + (1.0 - a2) * nov * nov) + 1e-20); }

// Échantillonnage des normales visibles (Heitz 2018)
vec3 sampleVndf(vec3 v, float a, vec2 u)
{
    const vec3 vh = normalize(vec3(a * v.x, a * v.y, v.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const vec3 t1 = lensq > 0.0 ? vec3(-vh.y, vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    const vec3 t2 = cross(vh, t1);
    const float r = sqrt(u.x);
    const float phi = kTwoPi * u.y;
    const float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    const float s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
    const vec3 nh = t1 * p1 + t2 * p2 + vh * sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2));
    return normalize(vec3(a * nh.x, a * nh.y, max(1e-6, nh.z)));
}

struct BsdfLobes {
    vec3 f0;
    vec3 diffuse;
    float a2Spec, aSpec;
    float a2Coat, aCoat;
    float coat;
    float pDiff, pSpec, pCoat;
};

BsdfLobes makeLobes(Surface s, float nov)
{
    BsdfLobes l;
    const float f0d = 0.08 * s.specular;
    l.f0 = mix(vec3(f0d), s.base, s.metallic);
    l.diffuse = s.base * (1.0 - s.metallic);
    l.aSpec = max(s.roughness * s.roughness, 1e-3);
    l.a2Spec = l.aSpec * l.aSpec;
    l.aCoat = max(s.ccRoughness * s.ccRoughness, 1e-3);
    l.a2Coat = l.aCoat * l.aCoat;
    l.coat = s.clearcoat;
    const float es = luminance(schlick3(l.f0, nov));
    const float ed = luminance(l.diffuse) * (1.0 - es);
    const float ec = l.coat * schlick1(0.04, nov);
    const float sum = es + ed + ec;
    if (sum <= 0.0) {
        l.pDiff = 1.0;
        l.pSpec = 0.0;
        l.pCoat = 0.0;
    } else {
        l.pDiff = ed / sum;
        l.pSpec = es / sum;
        l.pCoat = ec / sum;
    }
    return l;
}

// wo, wi en repère local (z = normale)
vec3 evalBsdf(BsdfLobes l, vec3 wo, vec3 wi, out float pdf)
{
    pdf = 0.0;
    const float nov = wo.z, nol = wi.z;
    if (nov <= 0.0 || nol <= 0.0) return vec3(0.0);
    const vec3 h = normalize(wo + wi);
    const float noh = max(h.z, 0.0), voh = max(dot(wo, h), 0.0);
    const vec3 fs = schlick3(l.f0, voh) * (ggxD(noh, l.a2Spec) * smithG1(nov, l.a2Spec) * smithG1(nol, l.a2Spec) /
                                           (4.0 * nov * nol));
    const vec3 fd = l.diffuse * (kInvPi * (1.0 - luminance(schlick3(l.f0, nov))));
    const float fc = l.coat * schlick1(0.04, voh);
    const float coatSpec = fc * ggxD(noh, l.a2Coat) * smithG1(nov, l.a2Coat) * smithG1(nol, l.a2Coat) / (4.0 * nov * nol);
    pdf = l.pDiff * nol * kInvPi + l.pSpec * ggxD(noh, l.a2Spec) * smithG1(nov, l.a2Spec) / (4.0 * nov) +
          l.pCoat * ggxD(noh, l.a2Coat) * smithG1(nov, l.a2Coat) / (4.0 * nov);
    return (fd + fs) * (1.0 - fc) + vec3(coatSpec);
}

bool sampleBsdf(BsdfLobes l, vec3 wo, inout Rng rng, out vec3 wi, out vec3 f, out float pdf)
{
    const float u = rngNext(rng);
    const vec2 u2 = rngNext2(rng);
    if (u < l.pDiff) {
        const float r = sqrt(u2.x), phi = kTwoPi * u2.y;
        wi = vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u2.x)));
    } else {
        const float a = (u < l.pDiff + l.pSpec) ? l.aSpec : l.aCoat;
        const vec3 h = sampleVndf(wo, a, u2);
        wi = reflect(-wo, h);
    }
    f = vec3(0.0);
    pdf = 0.0;
    if (wi.z <= 0.0) return false;
    f = evalBsdf(l, wo, wi, pdf);
    return pdf > 1e-8;
}

// Fresnel diélectrique exact ; eta = n_incident / n_transmis
float fresnelDielectric(float cosi, float eta)
{
    const float sint2 = eta * eta * (1.0 - cosi * cosi);
    if (sint2 >= 1.0) return 1.0;
    const float cost = sqrt(1.0 - sint2);
    const float rs = (eta * cosi - cost) / (eta * cosi + cost);
    const float rp = (cosi - eta * cost) / (cosi + eta * cost);
    return 0.5 * (rs * rs + rp * rp);
}

vec3 beerLambert(vec3 colorAtDistance, float ratio) { return pow(max(colorAtDistance, vec3(1e-4)), vec3(ratio)); }

// Interface diélectrique lisse ou dépolie (GGX + VNDF, Walter et al. 2007)
bool sampleDielectric(vec3 rd, vec3 nOut, float ior, float roughness, inout Rng rng, out vec3 nd, out float weight,
                      out bool refracted)
{
    nd = vec3(0.0);
    weight = 0.0;
    refracted = false;
    const bool entering = dot(rd, nOut) < 0.0;
    const vec3 n = entering ? nOut : -nOut;
    const float eta = entering ? 1.0 / ior : ior;
    vec3 m = n;
    float a2 = 0.0;
    const vec2 u2 = rngNext2(rng);
    if (roughness > 0.02) {
        const float a = max(roughness * roughness, 1e-3);
        a2 = a * a;
        const Onb onb = makeOnb(n);
        vec3 wo = toLocal(onb, -rd);
        wo.z = max(wo.z, 1e-4);
        m = toWorld(onb, sampleVndf(normalize(wo), a, u2));
    }
    const float cosi = dot(-rd, m);
    if (cosi <= 0.0) return false;
    const float F = fresnelDielectric(min(cosi, 1.0), eta);
    refracted = rngNext(rng) >= F;
    if (!refracted) {
        nd = reflect(rd, m);
        if (dot(nd, n) <= 0.0) return false;
    } else {
        const float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
        nd = normalize(rd * eta + m * (eta * cosi - sqrt(max(k, 0.0))));
        if (dot(nd, n) >= 0.0) return false;
    }
    weight = a2 > 0.0 ? smithG1(abs(dot(nd, n)), a2) : 1.0;
    return true;
}

#endif
