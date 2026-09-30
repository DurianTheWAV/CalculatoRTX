// CalculatoRTX - environnement "studio" (identique à src/render/Kernels.cu).
#ifndef CRTX_ENV_GLSL
#define CRTX_ENV_GLSL

vec3 studio(vec3 d)
{
    const float y = d.y;
    const vec3 sky = mix(vec3(0.010, 0.011, 0.014), vec3(0.030, 0.034, 0.044), smoothstep(0.0, 0.8, y));
    const vec3 ground = vec3(0.006, 0.0055, 0.005);
    vec3 c = y > 0.0 ? sky : ground;
    c += vec3(0.018, 0.020, 0.026) * exp(-abs(y) * 9.0);  // halo d'horizon
    // grand réflecteur diffus au plafond (reflets doux sur les surfaces vernies)
    const float k1 = dot(d, normalize(vec3(-0.25, 0.92, 0.3)));
    c += vec3(0.20, 0.195, 0.185) * smoothstep(0.90, 0.975, k1);
    // bandeau vert à l'horizon arrière (clin d'œil "RTX")
    if (d.z < 0.0) {
        const float t = (y - 0.12) / 0.035;
        c += vec3(0.06, 0.30, 0.0) * (exp(-t * t) * smoothstep(0.2, 0.9, -d.z));
    }
    // fenêtre chaude latérale
    const float k2 = dot(d, normalize(vec3(0.95, 0.25, 0.1)));
    c += vec3(0.35, 0.24, 0.14) * smoothstep(0.965, 0.985, k2);
    return c;
}

#endif
