// CalculatoRTX - logique de saisie (aucun calcul numérique côté CPU).
#include "CalculatorController.h"

#include "../common/Check.h"
#include "../common/Glyphs.h"

#include <cstdio>

namespace crtx {

using namespace calc;

namespace {
std::string G(unsigned char c) { return std::string(1, static_cast<char>(c)); }
}  // namespace

CalculatorController::CalculatorController(CalcEngine& engine) : engine_(engine) {}

std::string CalculatorController::statusMessage(int status)
{
    switch (status) {
        case kSyntaxError: return "ERREUR SYNTAXE";
        case kDomainError: return "ERREUR MATH";
        case kDivByZero: return "DIVISION PAR 0";
        case kOverflow: return "DEPASSEMENT";
        case kStackError: return "PILE PLEINE";
        default: return "ERREUR";
    }
}

void CalculatorController::clearExpression()
{
    tokens_.clear();
    justEvaluated_ = false;
    errorText_.clear();
    previewText_.clear();
}

// Après "=", une nouvelle saisie repart de zéro (chiffre, fonction) ou continue avec ANS
// (opérateur, suffixe).
void CalculatorController::startNewIfEvaluated(bool continueWithAns)
{
    if (!justEvaluated_ && errorText_.empty()) return;
    const bool hadResult = justEvaluated_ && errorText_.empty();
    clearExpression();
    if (continueWithAns && hadResult) tokens_.push_back({TokType::Constant, "A", "ANS"});
}

bool CalculatorController::lastEndsValue() const
{
    if (tokens_.empty()) return false;
    const TokType t = tokens_.back().type;
    return t == TokType::Number || t == TokType::RParen || t == TokType::Postfix || t == TokType::Constant;
}

void CalculatorController::appendToken(TokType t, const std::string& prog, const std::string& disp)
{
    tokens_.push_back({t, prog, disp});
}

void CalculatorController::appendDigit(char c)
{
    startNewIfEvaluated(false);
    if (!lastIs(TokType::Number)) {
        if (c == '.') appendToken(TokType::Number, "0.", "0.");
        else appendToken(TokType::Number, std::string(1, c), std::string(1, c));
        return;
    }
    Token& t = tokens_.back();
    if (c == '.') {
        if (t.prog.find('.') != std::string::npos || t.prog.find('E') != std::string::npos) return;
    }
    if (t.prog.size() >= 40) return;
    t.prog += c;
    t.disp += c;
}

void CalculatorController::appendBinary(char prog, const std::string& disp)
{
    startNewIfEvaluated(true);
    if (tokens_.empty()) {
        if (prog == kOpSub) {
            appendToken(TokType::Negate, std::string(1, kOpNeg), G(glyph::kMinus));
            return;
        }
        tokens_.push_back({TokType::Constant, "A", "ANS"});
    }
    if (lastIs(TokType::Binary)) tokens_.pop_back();  // remplace l'opérateur précédent
    if (lastIs(TokType::Negate) || lastIs(TokType::Function) || lastIs(TokType::LParen)) {
        if (prog == kOpSub) appendToken(TokType::Negate, std::string(1, kOpNeg), G(glyph::kMinus));
        return;
    }
    appendToken(TokType::Binary, std::string(1, prog), disp);
}

void CalculatorController::appendPostfix(char prog, const std::string& disp)
{
    startNewIfEvaluated(true);
    if (tokens_.empty()) tokens_.push_back({TokType::Constant, "A", "ANS"});
    if (!lastEndsValue()) return;
    appendToken(TokType::Postfix, std::string(1, prog), disp);
}

void CalculatorController::toggleNegate()
{
    startNewIfEvaluated(true);
    if (lastIs(TokType::Number)) {
        Token& t = tokens_.back();
        const size_t e = t.prog.find('E');
        if (e != std::string::npos) {  // change le signe de l'exposant
            if (e + 1 < t.prog.size() && t.prog[e + 1] == '-') {
                t.prog.erase(e + 1, 1);
            } else {
                t.prog.insert(e + 1, "-");
            }
            // reconstruit l'affichage
            t.disp.clear();
            for (char c : t.prog) t.disp += (c == '-') ? G(glyph::kMinus) : std::string(1, c);
            return;
        }
        // signe du nombre : jeton Negate juste avant
        if (tokens_.size() >= 2 && tokens_[tokens_.size() - 2].type == TokType::Negate) {
            tokens_.erase(tokens_.end() - 2);
        } else {
            tokens_.insert(tokens_.end() - 1, Token{TokType::Negate, std::string(1, kOpNeg), G(glyph::kMinus)});
        }
        return;
    }
    if (lastEndsValue()) return;
    appendToken(TokType::Negate, std::string(1, kOpNeg), G(glyph::kMinus));
}

void CalculatorController::backspace()
{
    if (justEvaluated_ || !errorText_.empty()) {
        clearExpression();
        return;
    }
    if (tokens_.empty()) return;
    Token& t = tokens_.back();
    if (t.type == TokType::Number && t.prog.size() > 1) {
        const char removed = t.prog.back();
        t.prog.pop_back();
        // l'affichage peut contenir un glyphe multi-octet pour '-'
        if (removed == '-') t.disp.pop_back();
        else t.disp.pop_back();
        return;
    }
    tokens_.pop_back();
}

std::string CalculatorController::program() const
{
    std::string p;
    for (const Token& t : tokens_) p += t.prog;
    return p;
}

std::string CalculatorController::expressionText() const
{
    std::string s;
    for (const Token& t : tokens_) s += t.disp;
    return s;
}

void CalculatorController::updatePreview()
{
    previewText_.clear();
    if (tokens_.empty()) return;
    const Result r = engine_.evaluate(program(), angle_, ansHi_, ansLo_, memHi_, memLo_, &lastGpuMicros_);
    if (r.status == kOk) {
        previewText_ = r.text;
        consistent_ = r.consistent != 0;
    }
}

void CalculatorController::evaluate()
{
    if (tokens_.empty()) return;
    const std::string prog = program();
    const Result r = engine_.evaluate(prog, angle_, ansHi_, ansLo_, memHi_, memLo_, &lastGpuMicros_);
    evaluatedExpr_ = expressionText() + "=";
    if (r.status == kOk) {
        ansHi_ = r.ddHi;
        ansLo_ = r.ddLo;
        resultText_ = r.text;
        errorText_.clear();
        consistent_ = r.consistent != 0;
        CRTX_LOG("GPU : '%s' -> DD=%.17g%+.3g | FP64=%.17g | FP32=%.9g | intervalle=[%.17g, %.17g] | %s | %.1f us",
                 prog.c_str(), r.ddHi, r.ddLo, r.f64, static_cast<double>(r.f32), r.ivLo, r.ivHi,
                 r.consistent ? "valide" : "divergence", static_cast<double>(lastGpuMicros_));
    } else {
        errorText_ = statusMessage(r.status);
        CRTX_LOG("GPU : '%s' -> %s", prog.c_str(), errorText_.c_str());
    }
    justEvaluated_ = true;
    previewText_.clear();
}

void CalculatorController::memoryAdd(bool subtract)
{
    // M <- M +/- (expression courante ou ANS), l'addition est faite par le GPU.
    std::string expr = (tokens_.empty() || justEvaluated_) ? std::string("A") : program();
    const std::string prog = std::string("M") + (subtract ? "-" : "+") + "(" + expr + ")";
    const Result r = engine_.evaluate(prog, angle_, ansHi_, ansLo_, memHi_, memLo_, &lastGpuMicros_);
    if (r.status == kOk) {
        memHi_ = r.ddHi;
        memLo_ = r.ddLo;
        CRTX_LOG("GPU : mémoire = %.17g", r.ddHi + r.ddLo);
    } else {
        errorText_ = statusMessage(r.status);
        justEvaluated_ = true;
    }
}

void CalculatorController::press(KeyId key)
{
    const bool sh = shift_;
    if (key != KeyId::Second) shift_ = false;
    bool preview = true;

    switch (key) {
        case KeyId::Second: shift_ = !shift_; preview = false; break;
        case KeyId::D0: appendDigit('0'); break;
        case KeyId::D1: appendDigit('1'); break;
        case KeyId::D2: appendDigit('2'); break;
        case KeyId::D3: appendDigit('3'); break;
        case KeyId::D4: appendDigit('4'); break;
        case KeyId::D5: appendDigit('5'); break;
        case KeyId::D6: appendDigit('6'); break;
        case KeyId::D7: appendDigit('7'); break;
        case KeyId::D8: appendDigit('8'); break;
        case KeyId::D9: appendDigit('9'); break;
        case KeyId::Dot: appendDigit('.'); break;
        case KeyId::Exp:
            startNewIfEvaluated(false);
            if (lastIs(TokType::Number) && tokens_.back().prog.find('E') == std::string::npos) {
                tokens_.back().prog += 'E';
                tokens_.back().disp += 'E';
            } else if (!lastEndsValue()) {
                appendToken(TokType::Number, "1E", "1E");
            }
            break;
        case KeyId::Negate: toggleNegate(); break;
        case KeyId::Add: appendBinary(kOpAdd, "+"); break;
        case KeyId::Sub: appendBinary(kOpSub, G(glyph::kMinus)); break;
        case KeyId::Mul: appendBinary(kOpMul, G(glyph::kTimes)); break;
        case KeyId::Div: appendBinary(kOpDiv, G(glyph::kDivide)); break;
        case KeyId::Pow:
            if (sh) appendBinary(kOpRoot, "{x}" + G(glyph::kSqrt));
            else appendBinary(kOpPow, "^");
            break;
        case KeyId::Sin: case KeyId::Cos: case KeyId::Tan: case KeyId::Ln: case KeyId::Log: case KeyId::Sqrt: {
            startNewIfEvaluated(false);
            char op = 0;
            std::string d;
            const std::string m1 = "{" + G(glyph::kMinus) + "1}";
            switch (key) {
                case KeyId::Sin: op = sh ? kFnAsin : kFnSin; d = sh ? "sin" + m1 : "sin"; break;
                case KeyId::Cos: op = sh ? kFnAcos : kFnCos; d = sh ? "cos" + m1 : "cos"; break;
                case KeyId::Tan: op = sh ? kFnAtan : kFnTan; d = sh ? "tan" + m1 : "tan"; break;
                case KeyId::Ln: op = sh ? kFnExp : kFnLn; d = sh ? "e^" : "ln"; break;
                case KeyId::Log: op = sh ? kFnPow10 : kFnLog10; d = sh ? "10^" : "log"; break;
                default: op = sh ? kFnCbrt : kFnSqrt; d = sh ? "{3}" + G(glyph::kSqrt) : G(glyph::kSqrt); break;
            }
            appendToken(TokType::Function, std::string(1, op) + "(", d + "(");
            break;
        }
        case KeyId::LParen: startNewIfEvaluated(false); appendToken(TokType::LParen, "(", "("); break;
        case KeyId::RParen:
            if (!justEvaluated_) appendToken(TokType::RParen, ")", ")");
            break;
        case KeyId::Square:
            if (sh) appendPostfix(kPostCube, "{3}");
            else appendPostfix(kPostSquare, "{2}");
            break;
        case KeyId::Inverse: appendPostfix(kPostInv, "{" + G(glyph::kMinus) + "1}"); break;
        case KeyId::Fact: appendPostfix(kPostFact, "!"); break;
        case KeyId::Pi: startNewIfEvaluated(false); appendToken(TokType::Constant, "p", G(glyph::kPi)); break;
        case KeyId::Euler: startNewIfEvaluated(false); appendToken(TokType::Constant, "e", "e"); break;
        case KeyId::Ans: startNewIfEvaluated(false); appendToken(TokType::Constant, "A", "ANS"); break;
        case KeyId::MemRecall: startNewIfEvaluated(false); appendToken(TokType::Constant, "M", "M"); break;
        case KeyId::MemClear: memHi_ = memLo_ = 0.0; preview = false; break;
        case KeyId::MemPlus: memoryAdd(sh); preview = false; break;
        case KeyId::Drg: angle_ = angle_ == kDegrees ? kRadians : kDegrees; break;
        case KeyId::AllClear: clearExpression(); resultText_ = "0"; preview = false; break;
        case KeyId::Backspace: backspace(); break;
        case KeyId::Equals: evaluate(); preview = false; break;
        default: break;
    }
    if (preview && !justEvaluated_) updatePreview();
}

DisplayContent CalculatorController::display() const
{
    DisplayContent d;
    if (justEvaluated_) {
        d.expression = evaluatedExpr_;
        if (!errorText_.empty()) {
            d.main = errorText_;
            d.error = true;
        } else {
            d.main = resultText_;
        }
    } else {
        d.expression = expressionText();
        if (tokens_.empty()) {
            d.main = "0";
        } else if (!previewText_.empty()) {
            d.main = previewText_;
            d.mainIsPreview = true;
        } else {
            d.main = "";
        }
    }
    d.statusLeft = std::string(shift_ ? "2nd   " : "") + (angle_ == kDegrees ? "DEG" : "RAD");
    if (memHi_ != 0.0 || memLo_ != 0.0) d.statusLeft += "   M";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "GPU %.0fus  DD", static_cast<double>(lastGpuMicros_));
    d.statusRight = buf;
    if (consistent_) d.statusRight += " " + G(glyph::kCheck);
    return d;
}

}  // namespace crtx
