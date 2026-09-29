// CalculatoRTX - logique de saisie de la calculatrice.
//
// Cette classe ne fait que manipuler du texte (jetons saisis) : toutes les valeurs
// numériques sont calculées par le GPU via calc::CalcEngine (y compris l'addition
// mémoire M+/M- et l'aperçu en direct du résultat).
#pragma once

#include "../calc/CalcEngine.h"
#include "../scene/CalculatorScene.h"
#include "../scene/Keys.h"

#include <string>
#include <vector>

namespace crtx {

class CalculatorController {
public:
    explicit CalculatorController(calc::CalcEngine& engine);

    void press(KeyId key);

    bool shift() const { return shift_; }
    DisplayContent display() const;

private:
    enum class TokType { Number, Binary, Function, LParen, RParen, Postfix, Constant, Negate };
    struct Token {
        TokType type;
        std::string prog;  // octets du programme GPU
        std::string disp;  // texte affiché (glyphes)
    };

    void clearExpression();
    void startNewIfEvaluated(bool continueWithAns);
    void appendDigit(char c);
    void appendBinary(char prog, const std::string& disp);
    void appendPostfix(char prog, const std::string& disp);
    void appendToken(TokType t, const std::string& prog, const std::string& disp);
    void toggleNegate();
    void backspace();
    void evaluate();
    void memoryAdd(bool subtract);
    void updatePreview();
    std::string program() const;
    std::string expressionText() const;
    bool lastIs(TokType t) const { return !tokens_.empty() && tokens_.back().type == t; }
    bool lastEndsValue() const;
    static std::string statusMessage(int status);

    calc::CalcEngine& engine_;
    std::vector<Token> tokens_;
    bool shift_ = false;
    calc::AngleMode angle_ = calc::kDegrees;
    double ansHi_ = 0.0, ansLo_ = 0.0;
    double memHi_ = 0.0, memLo_ = 0.0;
    bool justEvaluated_ = false;
    std::string evaluatedExpr_;
    std::string resultText_ = "0";
    std::string previewText_;
    std::string errorText_;
    bool consistent_ = false;
    float lastGpuMicros_ = 0.0f;
};

}  // namespace crtx
