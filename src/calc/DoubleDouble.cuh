// CalculatoRTX - arithmétique "double-double" (~106 bits de mantisse, ~31 chiffres) sur GPU.
//
// Une valeur DD = hi + lo avec |lo| <= ulp(hi)/2. Les opérations s'appuient sur les
// transformations exactes TwoSum / TwoProd (FMA matériel FP64 des SM Ada).
// Toutes les fonctions sont exécutées sur la carte graphique.
#pragma once

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include <math.h>

#include <cmath>
#include <limits>

// Les fonctions sont compilées pour le GPU CUDA (et aussi pour le CPU afin de permettre
// les tests unitaires hôte, voir tests/CalcHostTest.cpp). Le backend Vulkan en utilise un
// portage GLSL fidèle : src/vk/shaders/Calc.comp.
#if defined(__CUDACC__)
#define CRTX_CALC_HD __host__ __device__ inline
#else
#define CRTX_CALC_HD inline
#endif

namespace crtx {
namespace dd {

CRTX_CALC_HD double qnan()
{
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(0x7ff8000000000000LL);
#else
    return std::numeric_limits<double>::quiet_NaN();
#endif
}

CRTX_CALC_HD double inf()
{
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(0x7ff0000000000000LL);
#else
    return std::numeric_limits<double>::infinity();
#endif
}

// Tests portables (indépendants de <cmath> hôte / CUDA)
CRTX_CALC_HD bool isNaN(double x) { return x != x; }
CRTX_CALC_HD bool isFin(double x) { return (x - x) == 0.0; }

struct DD {
    double hi;
    double lo;
};

CRTX_CALC_HD DD make(double hi, double lo = 0.0) { return DD{hi, lo}; }

// ---- constantes (calculées avec mpmath, 300 bits)
CRTX_CALC_HD DD kPi() { return make(3.141592653589793, 1.2246467991473532e-16); }
CRTX_CALC_HD DD kPi2() { return make(1.5707963267948966, 6.123233995736766e-17); }
CRTX_CALC_HD DD kE() { return make(2.718281828459045, 1.4456468917292502e-16); }
CRTX_CALC_HD DD kLn2() { return make(0.6931471805599453, 2.3190468138462996e-17); }
CRTX_CALC_HD DD kLn10() { return make(2.302585092994046, -2.1707562233822494e-16); }
CRTX_CALC_HD DD kDeg2Rad() { return make(0.017453292519943295, 2.9486522708701687e-19); }
CRTX_CALC_HD DD kRad2Deg() { return make(57.29577951308232, -1.9878495670576283e-15); }

// ---- transformations exactes
CRTX_CALC_HD DD quickTwoSum(double a, double b)
{
    const double s = a + b;
    const double e = b - (s - a);
    return make(s, e);
}

CRTX_CALC_HD DD twoSum(double a, double b)
{
    const double s = a + b;
    const double bb = s - a;
    const double e = (a - (s - bb)) + (b - bb);
    return make(s, e);
}

CRTX_CALC_HD DD twoProd(double a, double b)
{
    const double p = a * b;
    const double e = fma(a, b, -p);
    return make(p, e);
}

// ---- opérations de base
CRTX_CALC_HD DD add(DD a, DD b)
{
    DD s = twoSum(a.hi, b.hi);
    DD t = twoSum(a.lo, b.lo);
    s.lo += t.hi;
    s = quickTwoSum(s.hi, s.lo);
    s.lo += t.lo;
    return quickTwoSum(s.hi, s.lo);
}

CRTX_CALC_HD DD neg(DD a) { return make(-a.hi, -a.lo); }
CRTX_CALC_HD DD sub(DD a, DD b) { return add(a, neg(b)); }

CRTX_CALC_HD DD mul(DD a, DD b)
{
    DD p = twoProd(a.hi, b.hi);
    p.lo = fma(a.hi, b.lo, fma(a.lo, b.hi, p.lo));
    return quickTwoSum(p.hi, p.lo);
}

CRTX_CALC_HD DD mulD(DD a, double b)
{
    DD p = twoProd(a.hi, b);
    p.lo = fma(a.lo, b, p.lo);
    return quickTwoSum(p.hi, p.lo);
}

CRTX_CALC_HD DD div(DD a, DD b)
{
    const double q1 = a.hi / b.hi;
    DD r = sub(a, mulD(b, q1));
    const double q2 = r.hi / b.hi;
    r = sub(r, mulD(b, q2));
    const double q3 = r.hi / b.hi;
    DD q = quickTwoSum(q1, q2);
    return add(q, make(q3));
}

CRTX_CALC_HD DD ldexpDD(DD a, int e) { return make(ldexp(a.hi, e), ldexp(a.lo, e)); }

CRTX_CALC_HD bool isZero(DD a) { return a.hi == 0.0; }
CRTX_CALC_HD bool isFinite(DD a) { return isFin(a.hi) && isFin(a.lo); }
CRTX_CALC_HD bool lt(DD a, DD b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
CRTX_CALC_HD bool eq(DD a, DD b) { return a.hi == b.hi && a.lo == b.lo; }
CRTX_CALC_HD DD absDD(DD a) { return a.hi < 0.0 ? neg(a) : a; }

CRTX_CALC_HD DD floorDD(DD a)
{
    double hi = floor(a.hi);
    double lo = 0.0;
    if (hi == a.hi) {  // hi entier : la partie fractionnaire est dans lo
        lo = floor(a.lo);
        return quickTwoSum(hi, lo);
    }
    return make(hi, lo);
}

CRTX_CALC_HD DD roundDD(DD a) { return floorDD(add(a, make(0.5))); }

CRTX_CALC_HD bool isInteger(DD a) { return isFinite(a) && eq(floorDD(a), a); }

// ---- racines
CRTX_CALC_HD DD sqrtDD(DD a)
{
    if (a.hi <= 0.0) return make(a.hi == 0.0 ? 0.0 : qnan());
    const double x = sqrt(a.hi);
    // x + (a - x^2) / (2x)
    DD r = sub(a, twoProd(x, x));
    return quickTwoSum(x, r.hi / (2.0 * x));
}

CRTX_CALC_HD DD cbrtDD(DD a)
{
    if (a.hi == 0.0) return make(0.0);
    const bool negative = a.hi < 0.0;
    DD x = absDD(a);
    DD y = make(cbrt(x.hi));
    // Newton : y <- y - (y^3 - x) / (3 y^2)   (deux itérations pour ~32 chiffres)
    for (int i = 0; i < 2; ++i) {
        DD y2 = mul(y, y);
        DD num = sub(mul(y2, y), x);
        y = sub(y, div(num, mulD(y2, 3.0)));
    }
    return negative ? neg(y) : y;
}

// ---- exponentielle / logarithme
CRTX_CALC_HD DD expDD(DD a)
{
    if (a.hi > 709.78) return make(inf());
    if (a.hi < -745.0) return make(0.0);
    if (a.hi == 0.0) return make(1.0);

    const double k = nearbyint(a.hi / kLn2().hi);
    DD r = sub(a, mulD(kLn2(), k));
    // réduction supplémentaire r / 2^10, série de Taylor de expm1, puis 10 doublements
    r = ldexpDD(r, -10);
    DD term = r;
    DD s = r;
    for (int i = 2; i < 14; ++i) {
        term = mul(term, r);
        term = div(term, make(static_cast<double>(i)));
        s = add(s, term);
        if (fabs(term.hi) < 1e-36) break;
    }
    // expm1(2x) = 2 expm1(x) + expm1(x)^2
    for (int i = 0; i < 10; ++i) s = add(mulD(s, 2.0), mul(s, s));
    s = add(s, make(1.0));
    return ldexpDD(s, static_cast<int>(k));
}

CRTX_CALC_HD DD logDD(DD a)
{
    if (a.hi <= 0.0) return make(qnan());
    if (a.hi == 1.0 && a.lo == 0.0) return make(0.0);
    // Newton sur exp : x <- x + a * exp(-x) - 1
    DD x = make(log(a.hi));
    for (int i = 0; i < 2; ++i) x = sub(add(x, mul(a, expDD(neg(x)))), make(1.0));
    return x;
}

// Puissance entière exacte (exponentiation binaire)
CRTX_CALC_HD DD powInt(DD x, long long n)
{
    bool inv = n < 0;
    unsigned long long m = inv ? static_cast<unsigned long long>(-n) : static_cast<unsigned long long>(n);
    DD r = make(1.0);
    DD b = x;
    while (m) {
        if (m & 1ull) r = mul(r, b);
        m >>= 1;
        if (m) b = mul(b, b);
    }
    return inv ? div(make(1.0), r) : r;
}

// ---- trigonométrie (argument en radians)
// Séries de Taylor sur |r| <= pi/4.
CRTX_CALC_HD void sinCosTaylor(DD r, DD& s, DD& c)
{
    const DD r2 = mul(r, r);
    // sin
    DD term = r;
    s = r;
    for (int k = 1; k < 18; ++k) {
        term = mul(term, r2);
        term = div(term, make(static_cast<double>((2 * k) * (2 * k + 1))));
        term = neg(term);
        s = add(s, term);
        if (fabs(term.hi) < 1e-36) break;
    }
    // cos
    term = make(1.0);
    c = make(1.0);
    for (int k = 1; k < 18; ++k) {
        term = mul(term, r2);
        term = div(term, make(static_cast<double>((2 * k - 1) * (2 * k))));
        term = neg(term);
        c = add(c, term);
        if (fabs(term.hi) < 1e-36) break;
    }
}

CRTX_CALC_HD void sinCosDD(DD a, DD& s, DD& c)
{
    const double k = nearbyint(a.hi / kPi2().hi);
    DD r = sub(a, mulD(kPi2(), k));
    DD sr, cr;
    sinCosTaylor(r, sr, cr);
    long long q = static_cast<long long>(k) & 3ll;
    if (q < 0) q += 4;
    switch (q) {
        case 0: s = sr; c = cr; break;
        case 1: s = cr; c = neg(sr); break;
        case 2: s = neg(sr); c = neg(cr); break;
        default: s = neg(cr); c = sr; break;
    }
}

// atan par Newton sur tan : z <- z - (sin z - x cos z) cos z
CRTX_CALC_HD DD atanDD(DD x)
{
    if (x.hi == 0.0) return make(0.0);
    DD z = make(atan(x.hi));
    for (int i = 0; i < 2; ++i) {
        DD s, c;
        sinCosDD(z, s, c);
        z = sub(z, mul(sub(s, mul(x, c)), c));
    }
    return z;
}

CRTX_CALC_HD DD asinDD(DD x)
{
    const DD one = make(1.0);
    const DD ax = absDD(x);
    if (lt(one, ax)) return make(qnan());
    if (eq(ax, one)) return x.hi > 0 ? kPi2() : neg(kPi2());
    return atanDD(div(x, sqrtDD(sub(one, mul(x, x)))));
}

CRTX_CALC_HD DD acosDD(DD x)
{
    const DD a = asinDD(x);
    if (!isFinite(a)) return a;
    return sub(kPi2(), a);
}

// ---- fonction Gamma en double-double (~31 chiffres), identique en GLSL (CalcDD.glsl)
// ln Gamma(z) par la série de Stirling (13 termes de Bernoulli) pour z >= 25.
CRTX_CALC_HD DD lgammaStirling(DD z)
{
    const double num[13] = {1.0, -1.0, 1.0, -1.0, 1.0, -691.0, 1.0, -3617.0, 43867.0, -174611.0, 77683.0,
                            -236364091.0, 657931.0};
    const double den[13] = {12.0, 360.0, 1260.0, 1680.0, 1188.0, 360360.0, 156.0, 122400.0, 244188.0, 125400.0,
                            5796.0, 1506960.0, 300.0};
    const DD lnz = logDD(z);
    DD r = sub(mul(sub(z, make(0.5)), lnz), z);
    r = add(r, mulD(logDD(mulD(kPi(), 2.0)), 0.5));  // + ln(2 pi) / 2
    const DD z2 = mul(z, z);
    DD t = div(make(1.0), z);
    for (int k = 0; k < 13; ++k) {
        r = add(r, mul(div(make(num[k]), make(den[k])), t));
        t = div(t, z2);
    }
    return r;
}

// Gamma(x) pour x >= 1/2 : récurrence jusqu'à x + n >= 25, puis Stirling
CRTX_CALC_HD DD gammaPositive(DD x)
{
    DD prod = make(1.0);
    DD z = x;
    while (z.hi < 25.0) {
        prod = mul(prod, z);
        z = add(z, make(1.0));
    }
    const DD g = expDD(lgammaStirling(z));
    if (!isFinite(g)) return g;
    return div(g, prod);
}

// Gamma(x), x non entier négatif ou nul (pôles traités par l'appelant) ; formule des
// compléments pour x < 1/2 : Gamma(x) = pi / (sin(pi x) Gamma(1 - x)).
CRTX_CALC_HD DD gammaDD(DD x)
{
    if (!isFinite(x)) return make(qnan());
    if (x.hi >= 0.5) return gammaPositive(x);
    if (x.hi < -200.0) return make(0.0);
    const DD g = gammaPositive(sub(make(1.0), x));
    if (!isFinite(g)) return make(0.0);
    DD s, c;
    sinCosDD(mul(kPi(), x), s, c);
    return div(kPi(), mul(s, g));
}

}  // namespace dd
}  // namespace crtx
