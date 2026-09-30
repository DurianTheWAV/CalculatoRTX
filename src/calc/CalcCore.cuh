// CalculatoRTX - cœur du moteur de calcul : analyse, évaluation multi-précision et
// formatage. Compilé pour le GPU (kernel calcKernel) et, pour les tests unitaires,
// également pour le CPU (tests/CalcHostTest.cu).
#pragma once

#include "CalcTypes.h"
#include "DoubleDouble.cuh"

#include "../common/Glyphs.h"

#include <cfloat>
#include <cmath>

namespace crtx {
namespace calc {
namespace core {

using dd::DD;

constexpr double kPiD = 3.141592653589793;

// ---- opérations à arrondi dirigé (intrinsèques matérielles FP64 sur le GPU)
#if defined(__CUDA_ARCH__)
CRTX_CALC_HD double iv_add_rd(double a, double b) { return __dadd_rd(a, b); }
CRTX_CALC_HD double iv_add_ru(double a, double b) { return __dadd_ru(a, b); }
CRTX_CALC_HD double iv_sub_rd(double a, double b) { return __dsub_rd(a, b); }
CRTX_CALC_HD double iv_sub_ru(double a, double b) { return __dsub_ru(a, b); }
CRTX_CALC_HD double iv_mul_rd(double a, double b) { return __dmul_rd(a, b); }
CRTX_CALC_HD double iv_mul_ru(double a, double b) { return __dmul_ru(a, b); }
CRTX_CALC_HD double iv_div_rd(double a, double b) { return __ddiv_rd(a, b); }
CRTX_CALC_HD double iv_div_ru(double a, double b) { return __ddiv_ru(a, b); }
CRTX_CALC_HD double iv_sqrt_rd(double a) { return __dsqrt_rd(a); }
CRTX_CALC_HD double iv_sqrt_ru(double a) { return __dsqrt_ru(a); }
CRTX_CALC_HD double ivExp10(double a) { return exp10(a); }
#else
// Émulation CPU (tests) des arrondis dirigés : arrondi au plus proche, puis le signe de
// l'erreur exacte (TwoSum / TwoProd / reste) indique s'il faut passer au double adjacent.
// Un résultat exact reste exact, comme avec les instructions matérielles __dadd_rd...
inline double ivDown(double v) { return std::nextafter(v, -std::numeric_limits<double>::infinity()); }
inline double ivUp(double v) { return std::nextafter(v, std::numeric_limits<double>::infinity()); }
inline double iv_add_rd(double a, double b) { const dd::DD s = dd::twoSum(a, b); return s.lo < 0.0 ? ivDown(s.hi) : s.hi; }
inline double iv_add_ru(double a, double b) { const dd::DD s = dd::twoSum(a, b); return s.lo > 0.0 ? ivUp(s.hi) : s.hi; }
inline double iv_sub_rd(double a, double b) { return iv_add_rd(a, -b); }
inline double iv_sub_ru(double a, double b) { return iv_add_ru(a, -b); }
inline double iv_mul_rd(double a, double b) { const dd::DD p = dd::twoProd(a, b); return p.lo < 0.0 ? ivDown(p.hi) : p.hi; }
inline double iv_mul_ru(double a, double b) { const dd::DD p = dd::twoProd(a, b); return p.lo > 0.0 ? ivUp(p.hi) : p.hi; }
inline double ivDivRem(double a, double b, double q)  // signe de a/b - q
{
    const dd::DD p = dd::twoProd(q, b);
    const double r = (a - p.hi) - p.lo;
    return b > 0.0 ? r : -r;
}
inline double iv_div_rd(double a, double b) { const double q = a / b; return ivDivRem(a, b, q) < 0.0 ? ivDown(q) : q; }
inline double iv_div_ru(double a, double b) { const double q = a / b; return ivDivRem(a, b, q) > 0.0 ? ivUp(q) : q; }
inline double iv_sqrt_rd(double a)
{
    const double s = std::sqrt(a);
    const dd::DD p = dd::twoProd(s, s);
    return (a - p.hi) - p.lo < 0.0 ? ivDown(s) : s;
}
inline double iv_sqrt_ru(double a)
{
    const double s = std::sqrt(a);
    const dd::DD p = dd::twoProd(s, s);
    return (a - p.hi) - p.lo > 0.0 ? ivUp(s) : s;
}
inline double ivExp10(double a) { return std::pow(10.0, a); }
#endif


constexpr int kMaxRpn = 128;
constexpr int kMaxStack = 64;

enum ItemKind : unsigned char { kValue = 0, kUnary = 1, kBinary = 2 };

struct RpnItem {
    unsigned char kind;
    char op;
    DD value;
};

// ============================================================================
//                         Analyse (voie 0 uniquement)
// ============================================================================

CRTX_CALC_HD bool isDigit(char c) { return c >= '0' && c <= '9'; }

CRTX_CALC_HD bool isPrefixFn(char c)
{
    switch (c) {
        case kFnSin: case kFnCos: case kFnTan: case kFnAsin: case kFnAcos: case kFnAtan:
        case kFnLn: case kFnLog10: case kFnSqrt: case kFnCbrt: case kFnExp: case kFnPow10:
            return true;
        default:
            return false;
    }
}

CRTX_CALC_HD bool isPostfix(char c)
{
    return c == kPostFact || c == kPostSquare || c == kPostCube || c == kPostInv;
}

CRTX_CALC_HD bool isBinary(char c)
{
    return c == kOpAdd || c == kOpSub || c == kOpMul || c == kOpDiv || c == kOpPow || c == kOpRoot;
}

CRTX_CALC_HD int precedence(char op)
{
    switch (op) {
        case kOpAdd: case kOpSub: return 1;
        case kOpMul: case kOpDiv: return 2;
        case kOpNeg: return 3;
        case kOpPow: case kOpRoot: return 4;
        default: return isPrefixFn(op) ? 6 : 0;
    }
}

CRTX_CALC_HD bool rightAssoc(char op) { return op == kOpPow || op == kOpRoot || op == kOpNeg; }

// Conversion d'un littéral décimal en double-double (exact jusqu'à ~31 chiffres).
CRTX_CALC_HD int parseNumber(const char* p, int len, int pos, DD& out)
{
    DD mant = dd::make(0.0);
    int decimals = 0;
    bool seenPoint = false;
    int digits = 0;
    int i = pos;
    for (; i < len; ++i) {
        const char c = p[i];
        if (isDigit(c)) {
            mant = dd::add(dd::mulD(mant, 10.0), dd::make(static_cast<double>(c - '0')));
            if (seenPoint) ++decimals;
            ++digits;
        } else if (c == '.' && !seenPoint) {
            seenPoint = true;
        } else {
            break;
        }
    }
    int expo = 0;
    if (i < len && p[i] == 'E') {
        ++i;
        bool negExp = false;
        if (i < len && (p[i] == '-' || p[i] == '+')) {
            negExp = p[i] == '-';
            ++i;
        }
        int e = 0;
        int ed = 0;
        while (i < len && isDigit(p[i]) && ed < 5) {
            e = e * 10 + (p[i] - '0');
            ++i;
            ++ed;
        }
        expo = negExp ? -e : e;
        if (digits == 0) mant = dd::make(1.0);  // "E5" seul = 1E5
    }
    const int p10 = expo - decimals;
    if (p10 >= 0)
        out = dd::mul(mant, dd::powInt(dd::make(10.0), p10));
    else
        out = dd::div(mant, dd::powInt(dd::make(10.0), -p10));
    return i;
}

struct Parser {
    RpnItem* out;
    int count;
    int status;

    CRTX_CALC_HD void emitValue(DD v)
    {
        if (count >= kMaxRpn) { status = kStackError; return; }
        out[count].kind = kValue;
        out[count].op = 0;
        out[count].value = v;
        ++count;
    }

    CRTX_CALC_HD void emitOp(char op)
    {
        if (count >= kMaxRpn) { status = kStackError; return; }
        out[count].kind = (isBinary(op)) ? kBinary : kUnary;
        out[count].op = op;
        out[count].value = dd::make(0.0);
        ++count;
    }
};

CRTX_CALC_HD int parseProgram(const Request& rq, RpnItem* rpn, int& status)
{
    Parser ps{rpn, 0, kOk};
    char ops[kMaxStack];
    int top = 0;
    bool prevValue = false;
    const char* p = rq.program;
    const int len = rq.length < kMaxProgram ? rq.length : kMaxProgram;

    auto pushOp = [&](char op) {
        if (top >= kMaxStack) { ps.status = kStackError; return; }
        ops[top++] = op;
    };
    // Multiplication implicite : "2π", "3(4+5)", "(1+2)(3+4)", "2sin(30)"
    auto implicitMul = [&]() {
        if (!prevValue) return;
        while (top > 0 && ops[top - 1] != kParOpen) {
            const char o2 = ops[top - 1];
            if (precedence(o2) >= precedence(kOpMul)) { ps.emitOp(o2); --top; }
            else break;
        }
        pushOp(kOpMul);
        prevValue = false;
    };

    int i = 0;
    while (i < len && ps.status == kOk) {
        const char c = p[i];
        if (isDigit(c) || c == '.' || c == 'E') {
            implicitMul();
            DD v;
            i = parseNumber(p, len, i, v);
            ps.emitValue(v);
            prevValue = true;
            continue;
        }
        if (c == kConstPi || c == kConstE || c == kRegAns || c == kRegMem) {
            implicitMul();
            DD v = c == kConstPi ? dd::kPi()
                 : c == kConstE  ? dd::kE()
                 : c == kRegAns  ? dd::make(rq.ansHi, rq.ansLo)
                                 : dd::make(rq.memHi, rq.memLo);
            ps.emitValue(v);
            prevValue = true;
            ++i;
            continue;
        }
        if (isPrefixFn(c)) {
            implicitMul();
            pushOp(c);
            prevValue = false;
            ++i;
            continue;
        }
        if (c == kParOpen) {
            implicitMul();
            pushOp(kParOpen);
            prevValue = false;
            ++i;
            continue;
        }
        if (c == kParClose) {
            bool found = false;
            while (top > 0) {
                const char o = ops[--top];
                if (o == kParOpen) { found = true; break; }
                ps.emitOp(o);
            }
            if (!found) { ps.status = kSyntaxError; break; }
            if (top > 0 && isPrefixFn(ops[top - 1])) ps.emitOp(ops[--top]);
            if (!prevValue) { ps.status = kSyntaxError; break; }  // "()"
            prevValue = true;
            ++i;
            continue;
        }
        if (isPostfix(c)) {
            if (!prevValue) { ps.status = kSyntaxError; break; }
            ps.emitOp(c);
            ++i;
            continue;
        }
        if (c == kOpNeg || ((c == kOpSub || c == kOpAdd) && !prevValue)) {
            if (c != kOpAdd) pushOp(kOpNeg);  // '+' unaire ignoré
            prevValue = false;
            ++i;
            continue;
        }
        if (isBinary(c)) {
            if (!prevValue) { ps.status = kSyntaxError; break; }
            const int p1 = precedence(c);
            while (top > 0 && ops[top - 1] != kParOpen) {
                const char o2 = ops[top - 1];
                const int p2 = precedence(o2);
                if (p2 > p1 || (p2 == p1 && !rightAssoc(c))) { ps.emitOp(o2); --top; }
                else break;
            }
            pushOp(c);
            prevValue = false;
            ++i;
            continue;
        }
        ps.status = kSyntaxError;  // octet inconnu
    }
    if (ps.status == kOk) {
        if (ps.count == 0) {
            ps.status = kEmpty;
        } else if (!prevValue) {
            ps.status = kSyntaxError;  // expression terminée par un opérateur
        } else {
            while (top > 0) {  // parenthèses non fermées : fermeture automatique
                const char o = ops[--top];
                if (o != kParOpen) ps.emitOp(o);
            }
        }
    }
    status = ps.status;
    return ps.count;
}

// ============================================================================
//                   Arithmétiques (une par voie du warp)
// ============================================================================

CRTX_CALC_HD void setErr(int& err, int e)
{
    if (err == kOk) err = e;
}

// Classement d'un résultat non fini : +-inf => dépassement, NaN => domaine.
CRTX_CALC_HD void classify(double approx, int& err)
{
    if (dd::isNaN(approx)) setErr(err, kDomainError);
    else if (!dd::isFin(approx)) setErr(err, kOverflow);
}

// --------------------------------------------------------------------------
// Voie 0 : double-double
// --------------------------------------------------------------------------
struct NumDD {
    using T = DD;
    CRTX_CALC_HD static T fromDD(DD v) { return v; }
    CRTX_CALC_HD static double approx(T v) { return v.hi + v.lo; }

    CRTX_CALC_HD static T check(T r, double approx, int& err)
    {
        if (!dd::isFinite(r)) {
            if (dd::isNaN(approx)) setErr(err, kDomainError);
            else if (!dd::isFin(approx) || fabs(approx) > 1e300) setErr(err, kOverflow);
            else setErr(err, kDomainError);
        }
        return r;
    }

    // Réduction en degrés avec valeurs exactes aux multiples de 30° / 45°.
    CRTX_CALC_HD static bool exactDegTrig(char fn, DD x, DD& out, int& err)
    {
        DD r = dd::sub(x, dd::mulD(dd::roundDD(dd::div(x, dd::make(360.0))), 360.0));  // [-180,180]
        if (!dd::isInteger(r)) return false;
        const int deg = static_cast<int>(r.hi);
        const int a = ((deg % 360) + 360) % 360;
        double s = 2.0, c = 2.0;  // 2 = pas de valeur exacte
        switch (a) {
            case 0: s = 0.0; c = 1.0; break;
            case 30: s = 0.5; break;
            case 60: c = 0.5; break;
            case 90: s = 1.0; c = 0.0; break;
            case 120: c = -0.5; break;
            case 150: s = 0.5; break;
            case 180: s = 0.0; c = -1.0; break;
            case 210: s = -0.5; break;
            case 240: c = -0.5; break;
            case 270: s = -1.0; c = 0.0; break;
            case 300: c = 0.5; break;
            case 330: s = -0.5; break;
            default: break;
        }
        if (fn == kFnSin && s != 2.0) { out = dd::make(s); return true; }
        if (fn == kFnCos && c != 2.0) { out = dd::make(c); return true; }
        if (fn == kFnTan) {
            if (a == 90 || a == 270) { setErr(err, kDomainError); out = dd::make(0.0); return true; }
            if (a == 0 || a == 180) { out = dd::make(0.0); return true; }
            if (a == 45 || a == 225) { out = dd::make(1.0); return true; }
            if (a == 135 || a == 315) { out = dd::make(-1.0); return true; }
        }
        return false;
    }

    CRTX_CALC_HD static T trig(char fn, T x, int angle, int& err)
    {
        if (fabs(x.hi) > 1e15) { setErr(err, kDomainError); return dd::make(0.0); }
        T out;
        if (angle == kDegrees && exactDegTrig(fn, x, out, err)) return out;
        T rad = x;
        if (angle == kDegrees) {
            T r = dd::sub(x, dd::mulD(dd::roundDD(dd::div(x, dd::make(360.0))), 360.0));
            rad = dd::mul(r, dd::kDeg2Rad());
        }
        T s, c;
        dd::sinCosDD(rad, s, c);
        if (fn == kFnSin) out = s;
        else if (fn == kFnCos) out = c;
        else {
            if (fabs(c.hi) < 1e-40) { setErr(err, kDomainError); return dd::make(0.0); }
            out = dd::div(s, c);
        }
        // sin(pi) en radians : le résidu ~1e-33 provient de l'erreur de pi en DD.
        if (angle == kRadians && fabs(x.hi) > 0.5 && fabs(out.hi) < 1e-29 * fabs(x.hi)) out = dd::make(0.0);
        return out;
    }

    CRTX_CALC_HD static T invTrig(char fn, T x, int angle, int& err)
    {
        T r;
        if (fn == kFnAtan) r = dd::atanDD(x);
        else {
            if (fabs(x.hi) > 1.0 || (fabs(x.hi) == 1.0 && x.hi * x.lo > 0.0)) {
                setErr(err, kDomainError);
                return dd::make(0.0);
            }
            r = fn == kFnAsin ? dd::asinDD(x) : dd::acosDD(x);
        }
        if (angle == kDegrees) r = dd::mul(r, dd::kRad2Deg());
        return r;
    }

    CRTX_CALC_HD static T power(T x, T y, int& err)
    {
        if (dd::isInteger(y) && fabs(y.hi) < 4.0e18) {
            const long long n = static_cast<long long>(y.hi) + static_cast<long long>(y.lo);
            if (dd::isZero(x) && n < 0) { setErr(err, kDivByZero); return dd::make(0.0); }
            return check(dd::powInt(x, n), pow(approx(x), approx(y)), err);
        }
        if (x.hi > 0.0) return check(dd::expDD(dd::mul(y, dd::logDD(x))), pow(approx(x), approx(y)), err);
        if (dd::isZero(x)) {
            if (y.hi > 0.0) return dd::make(0.0);
            setErr(err, kDivByZero);
            return dd::make(0.0);
        }
        setErr(err, kDomainError);  // base négative, exposant non entier
        return dd::make(0.0);
    }

    CRTX_CALC_HD static T root(T n, T x, int& err)  // racine n-ième de x
    {
        if (dd::isZero(n)) { setErr(err, kDivByZero); return dd::make(0.0); }
        if (x.hi < 0.0) {
            if (dd::isInteger(n)) {
                const long long k = static_cast<long long>(n.hi);
                if (k & 1ll) return dd::neg(power(dd::neg(x), dd::div(dd::make(1.0), n), err));
            }
            setErr(err, kDomainError);
            return dd::make(0.0);
        }
        return power(x, dd::div(dd::make(1.0), n), err);
    }

    CRTX_CALC_HD static T factorial(T x, int& err)
    {
        if (dd::isInteger(x)) {
            if (x.hi < 0.0) { setErr(err, kDomainError); return dd::make(0.0); }
            if (x.hi > 170.0) { setErr(err, kOverflow); return dd::make(0.0); }
            T r = dd::make(1.0);
            for (int k = 2; k <= static_cast<int>(x.hi); ++k) r = dd::mulD(r, static_cast<double>(k));
            return r;
        }
        // non entier : Gamma(x+1) en double-double
        const T g = dd::gammaDD(dd::add(x, dd::make(1.0)));
        if (!dd::isFinite(g)) setErr(err, kOverflow);
        return g;
    }

    CRTX_CALC_HD static T unary(char op, T x, int angle, int& err)
    {
        switch (op) {
            case kOpNeg: return dd::neg(x);
            case kFnSin: case kFnCos: case kFnTan: return trig(op, x, angle, err);
            case kFnAsin: case kFnAcos: case kFnAtan: return invTrig(op, x, angle, err);
            case kFnLn:
                if (x.hi <= 0.0) { setErr(err, kDomainError); return dd::make(0.0); }
                return dd::logDD(x);
            case kFnLog10:
                if (x.hi <= 0.0) { setErr(err, kDomainError); return dd::make(0.0); }
                return dd::div(dd::logDD(x), dd::kLn10());
            case kFnSqrt:
                if (x.hi < 0.0) { setErr(err, kDomainError); return dd::make(0.0); }
                return dd::sqrtDD(x);
            case kFnCbrt: return dd::cbrtDD(x);
            case kFnExp: return check(dd::expDD(x), exp(approx(x)), err);
            case kFnPow10: return power(dd::make(10.0), x, err);
            case kPostFact: return factorial(x, err);
            case kPostSquare: return check(dd::mul(x, x), approx(x) * approx(x), err);
            case kPostCube: return check(dd::mul(dd::mul(x, x), x), approx(x) * approx(x) * approx(x), err);
            case kPostInv:
                if (dd::isZero(x)) { setErr(err, kDivByZero); return dd::make(0.0); }
                return dd::div(dd::make(1.0), x);
            default: setErr(err, kSyntaxError); return x;
        }
    }

    CRTX_CALC_HD static T binary(char op, T a, T b, int& err)
    {
        switch (op) {
            case kOpAdd: return check(dd::add(a, b), approx(a) + approx(b), err);
            case kOpSub: return check(dd::sub(a, b), approx(a) - approx(b), err);
            case kOpMul: return check(dd::mul(a, b), approx(a) * approx(b), err);
            case kOpDiv:
                if (dd::isZero(b)) { setErr(err, kDivByZero); return dd::make(0.0); }
                return check(dd::div(a, b), approx(a) / approx(b), err);
            case kOpPow: return power(a, b, err);
            case kOpRoot: return root(a, b, err);
            default: setErr(err, kSyntaxError); return a;
        }
    }

    CRTX_CALC_HD static bool finite(T v) { return dd::isFinite(v); }
};

// --------------------------------------------------------------------------
// Voies 1 et 2 : virgule flottante IEEE native (FP64 / FP32)
// --------------------------------------------------------------------------
template <typename F>
struct NumFP {
    using T = F;
    CRTX_CALC_HD static T fromDD(DD v) { return static_cast<F>(v.hi + v.lo); }
    CRTX_CALC_HD static double approx(T v) { return static_cast<double>(v); }
    CRTX_CALC_HD static F pi() { return static_cast<F>(kPiD); }

    CRTX_CALC_HD static T trig(char fn, T x, int angle, int& err)
    {
        T r = angle == kDegrees ? x * (pi() / static_cast<F>(180)) : x;
        if (fn == kFnSin) return sin(r);
        if (fn == kFnCos) return cos(r);
        const T c = cos(r);
        if (c == static_cast<F>(0)) setErr(err, kDomainError);
        return sin(r) / c;
    }

    CRTX_CALC_HD static T unary(char op, T x, int angle, int& err)
    {
        const F one = static_cast<F>(1);
        const F r2d = static_cast<F>(180) / pi();
        switch (op) {
            case kOpNeg: return -x;
            case kFnSin: case kFnCos: case kFnTan: return trig(op, x, angle, err);
            case kFnAsin:
                if (fabs(x) > one) setErr(err, kDomainError);
                return angle == kDegrees ? asin(x) * r2d : asin(x);
            case kFnAcos:
                if (fabs(x) > one) setErr(err, kDomainError);
                return angle == kDegrees ? acos(x) * r2d : acos(x);
            case kFnAtan: return angle == kDegrees ? atan(x) * r2d : atan(x);
            case kFnLn: if (x <= 0) setErr(err, kDomainError); return log(x);
            case kFnLog10: if (x <= 0) setErr(err, kDomainError); return log10(x);
            case kFnSqrt: if (x < 0) setErr(err, kDomainError); return sqrt(x);
            case kFnCbrt: return cbrt(x);
            case kFnExp: return exp(x);
            case kFnPow10: return pow(static_cast<F>(10), x);
            case kPostFact:
                if (x < 0 && floor(x) == x) setErr(err, kDomainError);
                return tgamma(x + one);
            case kPostSquare: return x * x;
            case kPostCube: return x * x * x;
            case kPostInv: if (x == 0) setErr(err, kDivByZero); return one / x;
            default: setErr(err, kSyntaxError); return x;
        }
    }

    CRTX_CALC_HD static T binary(char op, T a, T b, int& err)
    {
        switch (op) {
            case kOpAdd: return a + b;
            case kOpSub: return a - b;
            case kOpMul: return a * b;
            case kOpDiv: if (b == 0) setErr(err, kDivByZero); return a / b;
            case kOpPow: return pow(a, b);
            case kOpRoot:
                if (b < 0 && floor(a) == a && fmod(a, static_cast<F>(2)) != 0) return -pow(-b, static_cast<F>(1) / a);
                return pow(b, static_cast<F>(1) / a);
            default: setErr(err, kSyntaxError); return a;
        }
    }

    CRTX_CALC_HD static bool finite(T v) { return dd::isFin(v); }
};

// --------------------------------------------------------------------------
// Voie 3 : arithmétique d'intervalles FP64 à arrondis dirigés (encadrement garanti
//          pour + - x / sqrt ; élargi de quelques ulps pour les fonctions de la libm CUDA)
// --------------------------------------------------------------------------
struct Iv {
    double lo, hi;
};

struct NumIv {
    using T = Iv;
    CRTX_CALC_HD static double dn(double v, int ulps) { return v - fabs(v) * (ulps + 1) * DBL_EPSILON - 4.9e-324; }
    CRTX_CALC_HD static double up(double v, int ulps) { return v + fabs(v) * (ulps + 1) * DBL_EPSILON + 4.9e-324; }
    CRTX_CALC_HD static T pt(double v) { return Iv{v, v}; }
    CRTX_CALC_HD static T fromDD(DD v) { return Iv{iv_add_rd(v.hi, v.lo), iv_add_ru(v.hi, v.lo)}; }
    CRTX_CALC_HD static double approx(T v) { return 0.5 * (v.lo + v.hi); }

    CRTX_CALC_HD static T add(T a, T b) { return Iv{iv_add_rd(a.lo, b.lo), iv_add_ru(a.hi, b.hi)}; }
    CRTX_CALC_HD static T sub(T a, T b) { return Iv{iv_sub_rd(a.lo, b.hi), iv_sub_ru(a.hi, b.lo)}; }
    CRTX_CALC_HD static T mul(T a, T b)
    {
        const double l = fmin(fmin(iv_mul_rd(a.lo, b.lo), iv_mul_rd(a.lo, b.hi)),
                              fmin(iv_mul_rd(a.hi, b.lo), iv_mul_rd(a.hi, b.hi)));
        const double h = fmax(fmax(iv_mul_ru(a.lo, b.lo), iv_mul_ru(a.lo, b.hi)),
                              fmax(iv_mul_ru(a.hi, b.lo), iv_mul_ru(a.hi, b.hi)));
        return Iv{l, h};
    }
    CRTX_CALC_HD static T div(T a, T b, int& err)
    {
        if (b.lo <= 0.0 && b.hi >= 0.0) { setErr(err, kDivByZero); return pt(0.0); }
        const double l = fmin(fmin(iv_div_rd(a.lo, b.lo), iv_div_rd(a.lo, b.hi)),
                              fmin(iv_div_rd(a.hi, b.lo), iv_div_rd(a.hi, b.hi)));
        const double h = fmax(fmax(iv_div_ru(a.lo, b.lo), iv_div_ru(a.lo, b.hi)),
                              fmax(iv_div_ru(a.hi, b.lo), iv_div_ru(a.hi, b.hi)));
        return Iv{l, h};
    }
    CRTX_CALC_HD static T deg2rad() { return Iv{0.017453292519943295, nextafter(0.017453292519943295, 1.0)}; }
    CRTX_CALC_HD static T rad2deg() { return Iv{nextafter(57.29577951308232, 0.0), 57.29577951308232}; }

    CRTX_CALC_HD static T sinCos(bool isSin, T r)
    {
        // Lipschitz 1 : f([m-h, m+h]) inclus dans [f(m)-h, f(m)+h]
        const double m = 0.5 * (r.lo + r.hi);
        const double h = iv_add_ru(iv_mul_ru(iv_sub_ru(r.hi, r.lo), 0.5), fabs(m) * 2.0 * DBL_EPSILON);
        const double v = isSin ? sin(m) : cos(m);
        return Iv{fmax(-1.0, dn(v, 2) - h), fmin(1.0, up(v, 2) + h)};
    }

    CRTX_CALC_HD static T unary(char op, T x, int angle, int& err)
    {
        const T r = angle == kDegrees ? mul(x, deg2rad()) : x;
        switch (op) {
            case kOpNeg: return Iv{-x.hi, -x.lo};
            case kFnSin: return sinCos(true, r);
            case kFnCos: return sinCos(false, r);
            case kFnTan: {
                const double k0 = floor(r.lo / kPiD - 0.5), k1 = floor(r.hi / kPiD - 0.5);
                if (k0 != k1) { setErr(err, kDomainError); return pt(0.0); }
                return Iv{dn(tan(r.lo), 2), up(tan(r.hi), 2)};
            }
            case kFnAsin: case kFnAcos: {
                if (x.lo < -1.0 || x.hi > 1.0) { setErr(err, kDomainError); return pt(0.0); }
                T v = op == kFnAsin ? Iv{dn(asin(x.lo), 2), up(asin(x.hi), 2)}
                                    : Iv{dn(acos(x.hi), 2), up(acos(x.lo), 2)};
                return angle == kDegrees ? mul(v, rad2deg()) : v;
            }
            case kFnAtan: {
                T v = Iv{dn(atan(x.lo), 2), up(atan(x.hi), 2)};
                return angle == kDegrees ? mul(v, rad2deg()) : v;
            }
            case kFnLn:
                if (x.lo <= 0.0) { setErr(err, kDomainError); return pt(0.0); }
                return Iv{dn(log(x.lo), 1), up(log(x.hi), 1)};
            case kFnLog10:
                if (x.lo <= 0.0) { setErr(err, kDomainError); return pt(0.0); }
                return Iv{dn(log10(x.lo), 2), up(log10(x.hi), 2)};
            case kFnSqrt:
                if (x.hi < 0.0) { setErr(err, kDomainError); return pt(0.0); }
                return Iv{iv_sqrt_rd(fmax(x.lo, 0.0)), iv_sqrt_ru(x.hi)};
            case kFnCbrt: return Iv{dn(cbrt(x.lo), 1), up(cbrt(x.hi), 1)};
            case kFnExp: return Iv{dn(exp(x.lo), 1), up(exp(x.hi), 1)};
            case kFnPow10: return Iv{dn(ivExp10(x.lo), 2), up(ivExp10(x.hi), 2)};
            case kPostFact: {
                if (x.lo == x.hi && floor(x.lo) == x.lo && x.lo >= 0.0 && x.lo <= 170.0) {
                    T p = pt(1.0);
                    for (int k = 2; k <= static_cast<int>(x.lo); ++k) p = mul(p, pt(static_cast<double>(k)));
                    return p;
                }
                if (x.lo < 0.0 && floor(x.lo) == x.lo) { setErr(err, kDomainError); return pt(0.0); }
                const double g = tgamma(approx(x) + 1.0);
                const double w = fabs(g) * 1e-12 + fabs(x.hi - x.lo) * fabs(g) * 10.0;
                return Iv{g - w, g + w};
            }
            case kPostSquare: {
                T s = mul(x, x);
                if (x.lo <= 0.0 && x.hi >= 0.0) s.lo = 0.0;
                return s;
            }
            case kPostCube: return mul(mul(x, x), x);
            case kPostInv: return div(pt(1.0), x, err);
            default: setErr(err, kSyntaxError); return x;
        }
    }

    CRTX_CALC_HD static T power(T a, T b, int& err)
    {
        if (b.lo == b.hi && floor(b.lo) == b.lo && fabs(b.lo) <= 1024.0) {
            const int n = static_cast<int>(fabs(b.lo));
            T r = pt(1.0);
            for (int k = 0; k < n; ++k) r = mul(r, a);
            if (n % 2 == 0 && a.lo <= 0.0 && a.hi >= 0.0) r.lo = fmax(r.lo, 0.0);
            return b.lo < 0.0 ? div(pt(1.0), r, err) : r;
        }
        if (a.lo <= 0.0) {
            if (a.lo == 0.0 && a.hi == 0.0 && b.lo > 0.0) return pt(0.0);
            setErr(err, kDomainError);
            return pt(0.0);
        }
        const T lg = Iv{dn(log(a.lo), 1), up(log(a.hi), 1)};
        const T e = mul(b, lg);
        return Iv{dn(exp(e.lo), 1), up(exp(e.hi), 1)};
    }

    CRTX_CALC_HD static T binary(char op, T a, T b, int& err)
    {
        switch (op) {
            case kOpAdd: return add(a, b);
            case kOpSub: return sub(a, b);
            case kOpMul: return mul(a, b);
            case kOpDiv: return div(a, b, err);
            case kOpPow: return power(a, b, err);
            case kOpRoot: {
                if (b.hi < 0.0 && a.lo == a.hi && floor(a.lo) == a.lo && fmod(a.lo, 2.0) != 0.0) {
                    T r = power(Iv{-b.hi, -b.lo}, div(pt(1.0), a, err), err);
                    return Iv{-r.hi, -r.lo};
                }
                return power(b, div(pt(1.0), a, err), err);
            }
            default: setErr(err, kSyntaxError); return a;
        }
    }

    CRTX_CALC_HD static bool finite(T v) { return dd::isFin(v.lo) && dd::isFin(v.hi); }
};

// ============================================================================
//                            Évaluation RPN générique
// ============================================================================
template <typename Num>
CRTX_CALC_HD int evaluateRpn(const RpnItem* rpn, int n, int angle, typename Num::T& result)
{
    using T = typename Num::T;
    T stack[kMaxStack];
    int sp = 0;
    int err = kOk;
    for (int i = 0; i < n && err == kOk; ++i) {
        const RpnItem& it = rpn[i];
        if (it.kind == kValue) {
            if (sp >= kMaxStack) { err = kStackError; break; }
            stack[sp++] = Num::fromDD(it.value);
        } else if (it.kind == kUnary) {
            if (sp < 1) { err = kSyntaxError; break; }
            stack[sp - 1] = Num::unary(it.op, stack[sp - 1], angle, err);
        } else {
            if (sp < 2) { err = kSyntaxError; break; }
            const T b = stack[--sp];
            stack[sp - 1] = Num::binary(it.op, stack[sp - 1], b, err);
        }
    }
    if (err == kOk && sp != 1) err = kSyntaxError;
    if (err == kOk) {
        result = stack[0];
        if (!Num::finite(result)) classify(Num::approx(result), err);
        if (err == kOk && !Num::finite(result)) err = kOverflow;
    }
    return err;
}

// ============================================================================
//                     Formatage décimal (double-double -> texte)
// ============================================================================
CRTX_CALC_HD void appendChar(char* s, int& n, char c)
{
    if (n < kMaxText - 1) s[n++] = c;
}

CRTX_CALC_HD void appendInt(char* s, int& n, int v)
{
    char tmp[12];
    int k = 0;
    if (v == 0) tmp[k++] = '0';
    while (v > 0) { tmp[k++] = static_cast<char>('0' + v % 10); v /= 10; }
    while (k > 0) appendChar(s, n, tmp[--k]);
}

// Extrait 'sig' chiffres significatifs arrondis de |v| ; retourne l'exposant décimal.
CRTX_CALC_HD int extractDigits(DD a, int sig, char* digits)
{
    int e = static_cast<int>(floor(log10(a.hi)));
    // N = round(a / 10^(e - sig + 1)), entier < 10^sig (exact en DD)
    for (int attempt = 0; attempt < 3; ++attempt) {
        const int shift = e - sig + 1;
        DD scaled;
        if (shift >= 0) {
            scaled = dd::div(a, dd::powInt(dd::make(10.0), shift));
        } else {  // en deux fois : 10^309 dépasserait la plage des doubles (résultats < 1e-299)
            const int k1 = -shift > 300 ? 300 : -shift;
            scaled = dd::mul(dd::mul(a, dd::powInt(dd::make(10.0), k1)), dd::powInt(dd::make(10.0), -shift - k1));
        }
        DD nr = dd::roundDD(scaled);
        double lim = 1.0;
        for (int k = 0; k < sig; ++k) lim *= 10.0;
        if (nr.hi >= lim) { ++e; continue; }       // l'arrondi a fait "déborder" (9.99 -> 10.0)
        if (nr.hi < lim / 10.0) { --e; continue; }  // estimation log10 trop grande
        unsigned long long u = static_cast<unsigned long long>(nr.hi);
        const long long lo = static_cast<long long>(nr.lo);
        u = static_cast<unsigned long long>(static_cast<long long>(u) + lo);
        for (int k = sig - 1; k >= 0; --k) {
            digits[k] = static_cast<char>('0' + u % 10ull);
            u /= 10ull;
        }
        return e;
    }
    for (int k = 0; k < sig; ++k) digits[k] = '0';
    digits[0] = '1';
    return e;
}

CRTX_CALC_HD void formatResult(DD v, char* out)
{
    int n = 0;
    if (v.hi == 0.0) {
        appendChar(out, n, '0');
        out[n] = 0;
        return;
    }
    if (v.hi < 0.0) {
        appendChar(out, n, static_cast<char>(glyph::kMinus));
        v = dd::neg(v);
    }
    const int e0 = static_cast<int>(floor(log10(v.hi)));
    char digits[20];
    const bool fixed = e0 >= -6 && e0 < 15;
    int sig = fixed ? (e0 >= 0 ? 15 : 15 + e0) : 10;
    if (sig < 1) sig = 1;
    int e = extractDigits(v, sig, digits);
    if (fixed && e >= 15) {  // l'arrondi a franchi 10^15 : notation scientifique
        sig = 10;
        e = extractDigits(v, sig, digits);
    }
    const bool useFixed = e >= -6 && e < 15;
    // supprime les zéros non significatifs en fin de mantisse
    int last = sig - 1;
    if (useFixed) {
        const int minKeep = e >= 0 ? e : 0;  // ne pas couper la partie entière
        while (last > minKeep && digits[last] == '0') --last;
    } else {
        while (last > 0 && digits[last] == '0') --last;
    }
    if (useFixed) {
        if (e < 0) {
            appendChar(out, n, '0');
            appendChar(out, n, '.');
            for (int k = 0; k < -e - 1; ++k) appendChar(out, n, '0');
            for (int k = 0; k <= last; ++k) appendChar(out, n, digits[k]);
        } else {
            for (int k = 0; k <= last || k <= e; ++k) {
                appendChar(out, n, k <= last ? digits[k] : '0');
                if (k == e && k < last) appendChar(out, n, '.');
            }
        }
    } else {
        appendChar(out, n, digits[0]);
        if (last > 0) {
            appendChar(out, n, '.');
            for (int k = 1; k <= last; ++k) appendChar(out, n, digits[k]);
        }
        appendChar(out, n, static_cast<char>(glyph::kTimes));
        appendChar(out, n, '1');
        appendChar(out, n, '0');
        appendChar(out, n, glyph::kSupBegin);
        if (e < 0) appendChar(out, n, static_cast<char>(glyph::kMinus));
        appendInt(out, n, e < 0 ? -e : e);
        appendChar(out, n, glyph::kSupEnd);
    }
    out[n] = 0;
}

// Assemble le résultat final à partir des quatre voies d'évaluation.
CRTX_CALC_HD void finalizeResult(Result& r, int status, int rpnCount, DD vDD, double f64, float f32, Iv iv,
                                 int st64, int st32, int stIv)
{
    r.status = status;
    r.rpnLength = rpnCount;
    r.ddHi = vDD.hi;
    r.ddLo = vDD.lo;
    r.f64 = f64;
    r.f32 = f32;
    r.ivLo = iv.lo;
    r.ivHi = iv.hi;
    r.laneStatus[0] = status;
    r.laneStatus[1] = st64;
    r.laneStatus[2] = st32;
    r.laneStatus[3] = stIv;
    const double x = vDD.hi + vDD.lo;
    const bool inInterval = stIv == kOk && x >= iv.lo && x <= iv.hi;
    const bool agree64 = st64 == kOk && fabs(f64 - x) <= 1e-12 * fmax(1.0, fabs(x));
    r.consistent = (status == kOk && inInterval && agree64) ? 1 : 0;
    if (status == kOk) formatResult(vDD, r.text);
    else r.text[0] = 0;
}

// Version séquentielle (une seule unité d'exécution) : utilisée par les tests CPU.
CRTX_CALC_HD void evaluateSequential(const Request& rq, Result& r)
{
    RpnItem rpn[kMaxRpn];
    int status = kOk;
    const int n = parseProgram(rq, rpn, status);
    DD vDD = dd::make(0.0);
    double v64 = 0.0;
    float v32 = 0.0f;
    Iv iv{0.0, 0.0};
    int st64 = status, st32 = status, stIv = status;
    if (status == kOk) {
        st64 = evaluateRpn<NumFP<double>>(rpn, n, rq.angleMode, v64);
        st32 = evaluateRpn<NumFP<float>>(rpn, n, rq.angleMode, v32);
        stIv = evaluateRpn<NumIv>(rpn, n, rq.angleMode, iv);
        status = evaluateRpn<NumDD>(rpn, n, rq.angleMode, vDD);
    }
    finalizeResult(r, status, n, vDD, v64, v32, iv, st64, st32, stIv);
}

}  // namespace core
}  // namespace calc
}  // namespace crtx
