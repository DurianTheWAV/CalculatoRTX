// CalculatoRTX - arithmétique double-double (~31 chiffres) et fonctions FP64 en GLSL.
//
// Portage fidèle de src/calc/DoubleDouble.cuh pour le backend Vulkan (AMD, Intel, NVIDIA).
// Différences imposées par Vulkan :
//   * GLSL n'a pas de fonctions transcendantes en double : exp, log, sin, atan... sont
//     calculées ici en double-double puis arrondies ;
//   * le produit exact TwoProd n'utilise pas fma() (Vulkan autorise un fma non fusionné)
//     mais le découpage de Veltkamp/Dekker, exact avec des opérations IEEE arrondies ;
//   * les valeurs spéciales (NaN, infini) sont testées sur les bits ;
//   * les variables "precise" interdisent toute contraction ou réassociation.
// Requiert shaderFloat64 et shaderInt64 (GL_ARB_gpu_shader_int64, déclaré par l'appelant).
#ifndef CRTX_CALC_DD_GLSL
#define CRTX_CALC_DD_GLSL

// ---------------------------------------------------------------- valeurs spéciales
double qnanD() { return packDouble2x32(uvec2(0u, 0x7ff80000u)); }
double infD() { return packDouble2x32(uvec2(0u, 0x7ff00000u)); }
bool isNaND(double x)
{
    const uvec2 b = unpackDouble2x32(x);
    return (b.y & 0x7ff00000u) == 0x7ff00000u && ((b.y & 0x000fffffu) != 0u || b.x != 0u);
}
bool isFinD(double x) { return (unpackDouble2x32(x).y & 0x7ff00000u) != 0x7ff00000u; }

// Double adjacent (émulation des arrondis dirigés de l'arithmétique d'intervalles)
double nextUpD(double x)
{
    if (isNaND(x) || x == infD()) return x;
    if (x == 0.0LF) return 2.2250738585072014e-308LF;  // plus petit normal (sûr même si les dénormaux sont vidés)
    uint64_t b = doubleBitsToUint64(x);
    b = x > 0.0LF ? b + 1ul : b - 1ul;
    return uint64BitsToDouble(b);
}
double nextDownD(double x) { return -nextUpD(-x); }

// ldexp en deux temps (le résultat intermédiaire reste représentable)
double ldexpD(double x, int e)
{
    const int h = e / 2;
    return ldexp(ldexp(x, h), e - h);
}

// ---------------------------------------------------------------- double-double (x = hi, y = lo)
const dvec2 kDdPi = dvec2(3.141592653589793LF, 1.2246467991473532e-16LF);
const dvec2 kDdPi2 = dvec2(1.5707963267948966LF, 6.123233995736766e-17LF);
const dvec2 kDdE = dvec2(2.718281828459045LF, 1.4456468917292502e-16LF);
const dvec2 kDdLn2 = dvec2(0.6931471805599453LF, 2.3190468138462996e-17LF);
const dvec2 kDdLn10 = dvec2(2.302585092994046LF, -2.1707562233822494e-16LF);
const dvec2 kDdDeg2Rad = dvec2(0.017453292519943295LF, 2.9486522708701687e-19LF);
const dvec2 kDdRad2Deg = dvec2(57.29577951308232LF, -1.9878495670576283e-15LF);

dvec2 ddMake(double hi) { return dvec2(hi, 0.0LF); }

dvec2 quickTwoSum(double a, double b)
{
    precise double s = a + b;
    precise double e = b - (s - a);
    return dvec2(s, e);
}

dvec2 twoSum(double a, double b)
{
    precise double s = a + b;
    precise double bb = s - a;
    precise double e = (a - (s - bb)) + (b - bb);
    return dvec2(s, e);
}

// Produit exact a*b = p + e sans fma (découpage de Veltkamp) ; les opérandes énormes sont
// d'abord ramenés par une puissance de 2 exacte pour que le découpage ne déborde pas.
dvec2 twoProd(double a, double b)
{
    precise double p = a * b;
    double sa = a, sb = b, scale = 1.0LF;
    if (abs(sa) > 6.0e299LF) {
        sa = ldexp(sa, -64);
        scale = 18446744073709551616.0LF;
    }
    if (abs(sb) > 6.0e299LF) {
        sb = ldexp(sb, -64);
        scale *= 18446744073709551616.0LF;
    }
    precise double ps = sa * sb;
    precise double ca = 134217729.0LF * sa;  // 2^27 + 1
    precise double ah = ca - (ca - sa);
    precise double al = sa - ah;
    precise double cb = 134217729.0LF * sb;
    precise double bh = cb - (cb - sb);
    precise double bl = sb - bh;
    precise double e = ((ah * bh - ps) + ah * bl + al * bh) + al * bl;
    return dvec2(p, e * scale);
}

dvec2 ddAdd(dvec2 a, dvec2 b)
{
    dvec2 s = twoSum(a.x, b.x);
    const dvec2 t = twoSum(a.y, b.y);
    precise double sl = s.y + t.x;
    s = quickTwoSum(s.x, sl);
    precise double sl2 = s.y + t.y;
    return quickTwoSum(s.x, sl2);
}

dvec2 ddNeg(dvec2 a) { return -a; }
dvec2 ddSub(dvec2 a, dvec2 b) { return ddAdd(a, -b); }

dvec2 ddMul(dvec2 a, dvec2 b)
{
    const dvec2 p = twoProd(a.x, b.x);
    precise double lo = p.y + (a.x * b.y + a.y * b.x);
    return quickTwoSum(p.x, lo);
}

dvec2 ddMulD(dvec2 a, double b)
{
    const dvec2 p = twoProd(a.x, b);
    precise double lo = p.y + a.y * b;
    return quickTwoSum(p.x, lo);
}

dvec2 ddDiv(dvec2 a, dvec2 b)
{
    const double q1 = a.x / b.x;
    dvec2 r = ddSub(a, ddMulD(b, q1));
    const double q2 = r.x / b.x;
    r = ddSub(r, ddMulD(b, q2));
    const double q3 = r.x / b.x;
    const dvec2 q = quickTwoSum(q1, q2);
    return ddAdd(q, ddMake(q3));
}

dvec2 ddLdexp(dvec2 a, int e) { return dvec2(ldexpD(a.x, e), ldexpD(a.y, e)); }

bool ddIsZero(dvec2 a) { return a.x == 0.0LF; }
bool ddIsFinite(dvec2 a) { return isFinD(a.x) && isFinD(a.y); }
bool ddLt(dvec2 a, dvec2 b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }
bool ddEq(dvec2 a, dvec2 b) { return a.x == b.x && a.y == b.y; }
dvec2 ddAbs(dvec2 a) { return a.x < 0.0LF ? -a : a; }

dvec2 ddFloor(dvec2 a)
{
    const double hi = floor(a.x);
    if (hi == a.x) return quickTwoSum(hi, floor(a.y));  // partie fractionnaire dans lo
    return dvec2(hi, 0.0LF);
}

dvec2 ddRound(dvec2 a) { return ddFloor(ddAdd(a, ddMake(0.5LF))); }
bool ddIsInteger(dvec2 a) { return ddIsFinite(a) && ddEq(ddFloor(a), a); }

// ---------------------------------------------------------------- racines
dvec2 ddSqrt(dvec2 a)
{
    if (a.x <= 0.0LF) return ddMake(a.x == 0.0LF ? 0.0LF : qnanD());
    const double x = sqrt(a.x);
    const dvec2 r = ddSub(a, twoProd(x, x));
    return quickTwoSum(x, r.x / (2.0LF * x));
}

double cbrtGuess(double x)  // x > 0 fini : estimation FP32 (~7 chiffres)
{
    int e;
    const double m = frexp(x, e);
    int q = e / 3;
    int r = e - 3 * q;
    if (r < 0) {
        r += 3;
        q -= 1;
    }
    return ldexpD(double(pow(float(ldexp(m, r)), 1.0 / 3.0)), q);
}

dvec2 ddCbrt(dvec2 a)
{
    if (a.x == 0.0LF) return ddMake(0.0LF);
    if (!isFinD(a.x)) return a;
    const bool negative = a.x < 0.0LF;
    const dvec2 x = ddAbs(a);
    dvec2 y = ddMake(cbrtGuess(x.x));
    for (int i = 0; i < 4; ++i) {  // Newton : y <- y - (y^3 - x) / (3 y^2)
        const dvec2 y2 = ddMul(y, y);
        const dvec2 num = ddSub(ddMul(y2, y), x);
        y = ddSub(y, ddDiv(num, ddMulD(y2, 3.0LF)));
    }
    return negative ? -y : y;
}

// ---------------------------------------------------------------- exponentielle / logarithme
dvec2 ddExp(dvec2 a)
{
    if (isNaND(a.x)) return a;
    if (a.x > 709.78LF) return ddMake(infD());
    if (a.x < -745.0LF) return ddMake(0.0LF);
    if (a.x == 0.0LF) return ddMake(1.0LF);
    const double k = roundEven(a.x / kDdLn2.x);
    dvec2 r = ddSub(a, ddMulD(kDdLn2, k));
    r = ddLdexp(r, -10);  // réduction r / 2^10, série de expm1, puis 10 doublements
    dvec2 term = r;
    dvec2 s = r;
    for (int i = 2; i < 14; ++i) {
        term = ddMul(term, r);
        term = ddDiv(term, ddMake(double(i)));
        s = ddAdd(s, term);
        if (abs(term.x) < 1e-36LF) break;
    }
    for (int i = 0; i < 10; ++i) s = ddAdd(ddMulD(s, 2.0LF), ddMul(s, s));  // expm1(2x)
    s = ddAdd(s, ddMake(1.0LF));
    return ddLdexp(s, int(k));
}

dvec2 ddLog(dvec2 a)
{
    if (a.x <= 0.0LF) return ddMake(a.x == 0.0LF ? -infD() : qnanD());
    if (!isFinD(a.x)) return a;
    if (a.x == 1.0LF && a.y == 0.0LF) return ddMake(0.0LF);
    int e;
    const double m = frexp(a.x, e);
    dvec2 x = ddMake(double(log(float(m))) + double(e) * kDdLn2.x);
    for (int i = 0; i < 8; ++i) {  // Newton sur exp : x <- x + a exp(-x) - 1
        const dvec2 dx = ddSub(ddMul(a, ddExp(-x)), ddMake(1.0LF));
        x = ddAdd(x, dx);
        if (abs(dx.x) <= 1e-33LF * max(1.0LF, abs(x.x))) break;
    }
    return x;
}

dvec2 ddPowInt(dvec2 x, int64_t n)
{
    const bool inv = n < 0l;
    uint64_t m = uint64_t(inv ? -n : n);
    dvec2 r = ddMake(1.0LF);
    dvec2 b = x;
    while (m != 0ul) {
        if ((m & 1ul) != 0ul) r = ddMul(r, b);
        m >>= 1;
        if (m != 0ul) b = ddMul(b, b);
    }
    return inv ? ddDiv(ddMake(1.0LF), r) : r;
}

// ---------------------------------------------------------------- trigonométrie (radians)
void ddSinCosTaylor(dvec2 r, out dvec2 s, out dvec2 c)
{
    const dvec2 r2 = ddMul(r, r);
    dvec2 term = r;
    s = r;
    for (int k = 1; k < 18; ++k) {
        term = ddMul(term, r2);
        term = -ddDiv(term, ddMake(double((2 * k) * (2 * k + 1))));
        s = ddAdd(s, term);
        if (abs(term.x) < 1e-36LF) break;
    }
    term = ddMake(1.0LF);
    c = ddMake(1.0LF);
    for (int k = 1; k < 18; ++k) {
        term = ddMul(term, r2);
        term = -ddDiv(term, ddMake(double((2 * k - 1) * (2 * k))));
        c = ddAdd(c, term);
        if (abs(term.x) < 1e-36LF) break;
    }
}

void ddSinCos(dvec2 a, out dvec2 s, out dvec2 c)
{
    const double k = roundEven(a.x / kDdPi2.x);
    const dvec2 r = ddSub(a, ddMulD(kDdPi2, k));
    dvec2 sr, cr;
    ddSinCosTaylor(r, sr, cr);
    const int q = int(int64_t(k) & 3l);
    if (q == 0) { s = sr; c = cr; }
    else if (q == 1) { s = cr; c = -sr; }
    else if (q == 2) { s = -sr; c = -cr; }
    else { s = -cr; c = sr; }
}

dvec2 ddAtan(dvec2 x)
{
    if (x.x == 0.0LF) return ddMake(0.0LF);
    if (!isFinD(x.x)) return isNaND(x.x) ? x : (x.x > 0.0LF ? kDdPi2 : -kDdPi2);
    dvec2 z = ddMake(double(atan(float(x.x))));
    for (int i = 0; i < 8; ++i) {  // Newton sur tan : z <- z - (sin z - x cos z) cos z
        dvec2 s, c;
        ddSinCos(z, s, c);
        const dvec2 dz = ddMul(ddSub(s, ddMul(x, c)), c);
        z = ddSub(z, dz);
        if (abs(dz.x) <= 1e-33LF * max(1.0LF, abs(z.x))) break;
    }
    return z;
}

dvec2 ddAsin(dvec2 x)
{
    const dvec2 one = ddMake(1.0LF);
    const dvec2 ax = ddAbs(x);
    if (ddLt(one, ax)) return ddMake(qnanD());
    if (ddEq(ax, one)) return x.x > 0.0LF ? kDdPi2 : -kDdPi2;
    return ddAtan(ddDiv(x, ddSqrt(ddSub(one, ddMul(x, x)))));
}

dvec2 ddAcos(dvec2 x)
{
    const dvec2 a = ddAsin(x);
    if (!ddIsFinite(a)) return a;
    return ddSub(kDdPi2, a);
}

// ---------------------------------------------------------------- fonctions FP64 (arrondi du résultat DD)
double f64Exp(double x) { return ddExp(ddMake(x)).x; }
double f64Log(double x) { return ddLog(ddMake(x)).x; }
double f64Log10(double x)
{
    if (x <= 0.0LF || !isFinD(x)) return f64Log(x);
    return ddDiv(ddLog(ddMake(x)), kDdLn10).x;
}
double f64Sin(double x)
{
    if (!isFinD(x) || abs(x) > 1e15LF) return qnanD();
    dvec2 s, c;
    ddSinCos(ddMake(x), s, c);
    return s.x;
}
double f64Cos(double x)
{
    if (!isFinD(x) || abs(x) > 1e15LF) return qnanD();
    dvec2 s, c;
    ddSinCos(ddMake(x), s, c);
    return c.x;
}
double f64Tan(double x)
{
    if (!isFinD(x) || abs(x) > 1e15LF) return qnanD();
    dvec2 s, c;
    ddSinCos(ddMake(x), s, c);
    return ddDiv(s, c).x;
}
double f64Atan(double x) { return ddAtan(ddMake(x)).x; }
double f64Asin(double x) { return ddAsin(ddMake(x)).x; }
double f64Acos(double x) { return ddAcos(ddMake(x)).x; }
double f64Cbrt(double x) { return ddCbrt(ddMake(x)).x; }

bool f64IsInt(double x) { return isFinD(x) && floor(x) == x; }
bool f64IsOdd(double x) { return f64IsInt(x) && abs(x) < 9.0e15LF && (int64_t(x) & 1l) != 0l; }

// pow(x, y) au sens de la libm C pour les cas usuels
double f64Pow(double x, double y)
{
    if (isNaND(x) || isNaND(y)) return qnanD();
    if (y == 0.0LF) return 1.0LF;
    if (x == 0.0LF) return y > 0.0LF ? 0.0LF : infD();
    if (x < 0.0LF) {
        if (!f64IsInt(y)) return qnanD();
        const double r = ddExp(ddMul(ddMake(y), ddLog(ddMake(-x)))).x;
        return f64IsOdd(y) ? -r : r;
    }
    return ddExp(ddMul(ddMake(y), ddLog(ddMake(x)))).x;
}

// ---------------------------------------------------------------- fonction Gamma (~31 chiffres)
// Même algorithme que gammaDD() de DoubleDouble.cuh : Stirling (13 termes) pour z >= 25,
// récurrence en dessous, formule des compléments pour x < 1/2.
dvec2 ddLgammaStirling(dvec2 z)
{
    const double num[13] = double[13](1.0LF, -1.0LF, 1.0LF, -1.0LF, 1.0LF, -691.0LF, 1.0LF, -3617.0LF, 43867.0LF,
                                      -174611.0LF, 77683.0LF, -236364091.0LF, 657931.0LF);
    const double den[13] = double[13](12.0LF, 360.0LF, 1260.0LF, 1680.0LF, 1188.0LF, 360360.0LF, 156.0LF, 122400.0LF,
                                      244188.0LF, 125400.0LF, 5796.0LF, 1506960.0LF, 300.0LF);
    const dvec2 lnz = ddLog(z);
    dvec2 r = ddSub(ddMul(ddSub(z, ddMake(0.5LF)), lnz), z);
    r = ddAdd(r, ddMulD(ddLog(ddMulD(kDdPi, 2.0LF)), 0.5LF));
    const dvec2 z2 = ddMul(z, z);
    dvec2 t = ddDiv(ddMake(1.0LF), z);
    for (int k = 0; k < 13; ++k) {
        r = ddAdd(r, ddMul(ddDiv(ddMake(num[k]), ddMake(den[k])), t));
        t = ddDiv(t, z2);
    }
    return r;
}

dvec2 ddGammaPositive(dvec2 x)
{
    dvec2 prod = ddMake(1.0LF);
    dvec2 z = x;
    while (z.x < 25.0LF) {
        prod = ddMul(prod, z);
        z = ddAdd(z, ddMake(1.0LF));
    }
    const dvec2 g = ddExp(ddLgammaStirling(z));
    if (!ddIsFinite(g)) return g;
    return ddDiv(g, prod);
}

dvec2 ddGamma(dvec2 x)
{
    if (!ddIsFinite(x)) return ddMake(qnanD());
    if (x.x >= 0.5LF) return ddGammaPositive(x);
    if (x.x < -200.0LF) return ddMake(0.0LF);
    const dvec2 g = ddGammaPositive(ddSub(ddMake(1.0LF), x));
    if (!ddIsFinite(g)) return ddMake(0.0LF);
    dvec2 s, c;
    ddSinCos(ddMul(kDdPi, x), s, c);
    return ddDiv(kDdPi, ddMul(s, g));
}

double f64Tgamma(double x)
{
    if (isNaND(x)) return x;
    if (f64IsInt(x) && x <= 0.0LF) return qnanD();
    if (x > 171.7LF) return infD();
    return ddGamma(ddMake(x)).x;
}

#endif
