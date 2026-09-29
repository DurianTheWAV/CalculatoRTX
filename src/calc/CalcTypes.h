// CalculatoRTX - structures d'échange hôte <-> GPU pour le moteur de calcul.
//
// L'hôte ne fait AUCUN calcul numérique : il transmet au GPU un "programme" (suite
// d'octets décrivant l'expression saisie). Le GPU l'analyse (tokenisation +
// shunting-yard), l'évalue en quatre arithmétiques en parallèle (double-double,
// FP64, FP32, intervalles FP64 à arrondi dirigé) et formate le résultat en texte.
#pragma once

namespace crtx {
namespace calc {

constexpr int kMaxProgram = 256;
constexpr int kMaxText = 48;

// ---- Opcodes du programme (octets) ----------------------------------------------
// Littéraux numériques : '0'..'9', '.', 'E' (exposant, suivi optionnellement de '-'/'+')
// Opérateurs binaires
constexpr char kOpAdd = '+';
constexpr char kOpSub = '-';  // binaire, ou unaire selon le contexte
constexpr char kOpMul = '*';
constexpr char kOpDiv = '/';
constexpr char kOpPow = '^';
constexpr char kOpRoot = 'R';  // a R b = racine a-ième de b
// Préfixes
constexpr char kOpNeg = '~';
constexpr char kFnSin = 's';
constexpr char kFnCos = 'c';
constexpr char kFnTan = 't';
constexpr char kFnAsin = 'S';
constexpr char kFnAcos = 'C';
constexpr char kFnAtan = 'T';
constexpr char kFnLn = 'l';
constexpr char kFnLog10 = 'L';
constexpr char kFnSqrt = 'q';
constexpr char kFnCbrt = 'Q';
constexpr char kFnExp = 'x';    // e^x
constexpr char kFnPow10 = 'X';  // 10^x
// Suffixes
constexpr char kPostFact = '!';
constexpr char kPostSquare = 'w';
constexpr char kPostCube = 'W';
constexpr char kPostInv = 'i';
// Constantes / registres
constexpr char kConstPi = 'p';
constexpr char kConstE = 'e';
constexpr char kRegAns = 'A';
constexpr char kRegMem = 'M';
// Parenthèses
constexpr char kParOpen = '(';
constexpr char kParClose = ')';

enum Status : int {
    kOk = 0,
    kSyntaxError = 1,
    kDomainError = 2,
    kDivByZero = 3,
    kOverflow = 4,
    kEmpty = 5,
    kStackError = 6,
};

enum AngleMode : int { kRadians = 0, kDegrees = 1 };

struct Request {
    char program[kMaxProgram];
    int length;
    int angleMode;
    double ansHi, ansLo;  // registre ANS (double-double)
    double memHi, memLo;  // registre mémoire (double-double)
};

struct Result {
    int status;
    int rpnLength;
    double ddHi, ddLo;     // résultat double-double (précision ~31 chiffres)
    double f64;            // même expression évaluée en FP64
    float f32;             // ... et en FP32
    double ivLo, ivHi;     // encadrement par arithmétique d'intervalles FP64
    int consistent;        // 1 si DD est dans l'intervalle et que FP64 concorde
    int laneStatus[4];     // statut par voie (DD, FP64, FP32, intervalle)
    char text[kMaxText];   // résultat formaté pour l'afficheur (généré sur GPU)
};

}  // namespace calc
}  // namespace crtx
