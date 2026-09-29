// CalculatoRTX - identifiants des touches de la calculatrice.
#pragma once

namespace crtx {

enum class KeyId : int {
    Second, Sin, Cos, Tan, Drg, D7, D8, D9, Div, AllClear,
    Square, Pow, Sqrt, Ln, Log, D4, D5, D6, Mul, Backspace,
    LParen, RParen, Pi, Euler, Fact, D1, D2, D3, Sub, Ans,
    MemClear, MemRecall, MemPlus, Inverse, Exp, D0, Dot, Negate, Add, Equals,
    Count
};

enum class KeyStyle { Number, Function, Operator, Equals, Clear, Shift };

}  // namespace crtx
