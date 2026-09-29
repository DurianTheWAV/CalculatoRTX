// CalculatoRTX - codes des caractères spéciaux utilisés par la police vectorielle 3D
// (partagés entre l'interface hôte, le formateur GPU et le générateur de géométrie).
#pragma once

namespace crtx {
namespace glyph {

// Caractères >= 0x80 : symboles mathématiques dessinés par la police vectorielle.
constexpr unsigned char kTimes = 0x80;      // ×
constexpr unsigned char kDivide = 0x81;     // ÷
constexpr unsigned char kMinus = 0x82;      // − (signe moins typographique)
constexpr unsigned char kSqrt = 0x83;       // √
constexpr unsigned char kPi = 0x84;         // π
constexpr unsigned char kPlusMinus = 0x85;  // ±
constexpr unsigned char kBackspace = 0x86;  // ⌫
constexpr unsigned char kCheck = 0x87;      // ✓
constexpr unsigned char kDegree = 0x88;     // °
constexpr unsigned char kBullet = 0x89;     // ·
constexpr unsigned char kArrowRight = 0x8A; // →

// Balises d'exposant (texte en indice supérieur) : "x{2}" dessine x²
constexpr char kSupBegin = '{';
constexpr char kSupEnd = '}';

}  // namespace glyph
}  // namespace crtx
