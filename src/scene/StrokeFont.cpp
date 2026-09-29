// CalculatoRTX - définition des glyphes de la police vectorielle.
//
// Mini-langage : unités de grille où la hauteur de capitale vaut 10 (ligne de base y=0,
// hauteur d'x = 7, jambage descendant jusqu'à -3). Une commande est une suite de points
// "x y" reliés, ';' commence un nouveau trait, "a cx cy rx ry deg0 deg1" insère un arc
// d'ellipse (sens trigonométrique si deg1 > deg0, horaire sinon).
#include "StrokeFont.h"

#include "../common/Glyphs.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace crtx {

namespace {
constexpr float kGrid = 10.0f;        // hauteur de capitale dans le mini-langage
constexpr float kSpacing = 1.7f;      // espace inter-caractères (unités de grille)
constexpr float kSupScale = 0.58f;    // taille des exposants
constexpr float kSupRaise = 5.6f;     // élévation des exposants (unités de grille)
}  // namespace

void StrokeFont::define(unsigned char c, float advance, const char* program)
{
    Glyph& g = glyphs_[c];
    g.advance = advance / kGrid;
    g.defined = true;
    g.strokes.clear();
    std::istringstream in(program);
    std::vector<float2> current;
    std::string tok;
    auto flush = [&]() {
        if (current.size() >= 2) g.strokes.push_back(current);
        current.clear();
    };
    while (in >> tok) {
        if (tok == ";") {
            flush();
        } else if (tok == "a") {
            float cx, cy, rx, ry, d0, d1;
            in >> cx >> cy >> rx >> ry >> d0 >> d1;
            const int n = std::max(4, static_cast<int>(std::fabs(d1 - d0) / 12.0f));
            for (int i = 0; i <= n; ++i) {
                const float t = (d0 + (d1 - d0) * static_cast<float>(i) / n) * 3.14159265f / 180.0f;
                current.push_back(make_float2((cx + rx * std::cos(t)) / kGrid, (cy + ry * std::sin(t)) / kGrid));
            }
        } else {
            const float x = std::strtof(tok.c_str(), nullptr);
            float y = 0.0f;
            in >> y;
            current.push_back(make_float2(x / kGrid, y / kGrid));
        }
    }
    flush();
}

StrokeFont::StrokeFont()
{
    // ---- chiffres (chasse fixe pour l'alignement des résultats)
    define('0', 6, "a 3 5 2.7 5 0 360");
    define('1', 6, "1.2 8 3.2 10 3.2 0 ; 1 0 5.4 0");
    define('2', 6, "a 3 7.2 2.8 2.8 165 -35 0.3 0 5.8 0");
    define('3', 6, "a 3 7.5 2.5 2.5 160 -90 ; a 3 2.6 2.9 2.6 90 -160");
    define('4', 6, "4.4 0 4.4 10 0 3.2 6 3.2");
    define('5', 6, "5.6 10 1 10 0.6 5.6 a 3 3.2 2.9 3.2 125 -145");
    define('6', 6, "4.6 10 0.6 3.6 ; a 3 3 2.6 3 0 360");
    define('7', 6, "0 10 6 10 2 0");
    define('8', 6, "a 3 7.6 2.3 2.4 0 360 ; a 3 2.7 2.8 2.7 0 360");
    define('9', 6, "1.4 0 5.4 6.4 ; a 3 7 2.6 3 0 360");

    // ---- capitales
    define('A', 6, "0 0 3 10 6 0 ; 1.1 3.4 4.9 3.4");
    define('B', 6, "0.4 0 0.4 10 3.2 10 a 3.2 7.6 2.4 2.4 90 -90 0.4 5.2 ; 0.4 5.2 3.4 5.2 a 3.4 2.6 2.6 2.6 90 -90 0.4 0");
    define('C', 6, "a 3.4 5 3.2 5 50 310");
    define('D', 6, "0.4 0 0.4 10 2.4 10 a 2.4 5 3.4 5 90 -90 0.4 0");
    define('E', 6, "5.6 10 0.4 10 0.4 0 5.6 0 ; 0.4 5.1 4.6 5.1");
    define('F', 6, "5.6 10 0.4 10 0.4 0 ; 0.4 5.1 4.6 5.1");
    define('G', 6, "a 3.4 5 3.2 5 50 330 6.2 4.4 3.6 4.4");
    define('H', 6, "0.4 0 0.4 10 ; 5.6 0 5.6 10 ; 0.4 5.1 5.6 5.1");
    define('I', 6, "3 0 3 10 ; 1.4 10 4.6 10 ; 1.4 0 4.6 0");
    define('J', 6, "5 10 5 3 a 2.6 3 2.4 3 0 -180");
    define('K', 6, "0.4 0 0.4 10 ; 5.8 10 0.4 3.6 ; 2.2 5.3 5.8 0");
    define('L', 6, "0.4 10 0.4 0 5.6 0");
    define('M', 7, "0.3 0 0.3 10 3.5 3.6 6.7 10 6.7 0");
    define('N', 6, "0.4 0 0.4 10 5.6 0 5.6 10");
    define('O', 6, "a 3 5 3 5 0 360");
    define('P', 6, "0.4 0 0.4 10 3.2 10 a 3.2 7.4 2.6 2.6 90 -90 0.4 4.8");
    define('Q', 6, "a 3 5 3 5 0 360 ; 3.6 2.4 6 -0.4");
    define('R', 6, "0.4 0 0.4 10 3.2 10 a 3.2 7.4 2.6 2.6 90 -90 0.4 4.8 ; 2.8 4.8 5.8 0");
    define('S', 6, "a 3 7.5 2.6 2.5 20 270 a 3 2.5 2.8 2.5 90 -160");
    define('T', 6, "0 10 6 10 ; 3 10 3 0");
    define('U', 6, "0.4 10 0.4 3 a 3 3 2.6 3 180 360 5.6 10");
    define('V', 6, "0 10 3 0 6 10");
    define('W', 7, "0 10 1.75 0 3.5 6.5 5.25 0 7 10");
    define('X', 6, "0.2 10 5.8 0 ; 0.2 0 5.8 10");
    define('Y', 6, "0 10 3 5 6 10 ; 3 5 3 0");
    define('Z', 6, "0.4 10 5.6 10 0.4 0 5.6 0");

    // ---- minuscules
    define('a', 6, "a 2.9 3.4 2.5 3.4 0 360 ; 5.4 7 5.4 0");
    define('b', 6, "0.6 10 0.6 0 ; a 3.1 3.4 2.5 3.4 0 360");
    define('c', 6, "a 3.2 3.5 2.8 3.5 45 315");
    define('d', 6, "5.4 10 5.4 0 ; a 2.9 3.4 2.5 3.4 0 360");
    define('e', 6, "0.5 3.6 5.5 3.6 a 3 3.5 2.5 3.5 0 310");
    define('f', 5, "a 3.6 8.2 1.4 1.8 30 180 2.2 0 ; 0.6 6.6 4.4 6.6");
    define('g', 6, "a 2.9 3.8 2.5 3.2 0 360 ; 5.4 7 5.4 -1 a 2.9 -1 2.5 2 0 -165");
    define('h', 6, "0.6 10 0.6 0 ; 0.6 4.2 a 3 4.2 2.4 2.8 180 0 5.4 0");
    define('i', 4, "2 0 2 7 ; 2 9.2 2 9.8");
    define('j', 5, "3.6 7 3.6 -1 a 1.8 -1 1.8 2 0 -150 ; 3.6 9.2 3.6 9.8");
    define('k', 6, "0.6 10 0.6 0 ; 5 7 0.6 2.8 ; 2.3 4.4 5.4 0");
    define('l', 4, "1.6 10 1.6 1.2 a 2.8 1.2 1.2 1.2 180 290");
    define('m', 7, "0.5 0 0.5 7 ; 0.5 5 a 2.1 5 1.6 2 180 0 3.7 0 ; 3.7 5 a 5.3 5 1.6 2 180 0 6.9 0");
    define('n', 6, "0.6 0 0.6 7 ; 0.6 4.2 a 3 4.2 2.4 2.8 180 0 5.4 0");
    define('o', 6, "a 3 3.5 2.6 3.5 0 360");
    define('p', 6, "0.6 7 0.6 -3 ; a 3.1 3.6 2.5 3.4 0 360");
    define('q', 6, "5.4 7 5.4 -3 ; a 2.9 3.6 2.5 3.4 0 360");
    define('r', 5, "0.8 0 0.8 7 ; 0.8 4 a 3.4 4 2.6 3 180 60");
    define('s', 6, "a 3 5.3 2.3 1.7 20 270 a 3 1.8 2.5 1.8 90 -160");
    define('t', 5, "2 9.4 2 1.4 a 3.4 1.4 1.4 1.4 180 290 ; 0.2 7 4.4 7");
    define('u', 6, "0.6 7 0.6 2.8 a 3 2.8 2.4 2.8 180 360 ; 5.4 7 5.4 0");
    define('v', 6, "0.2 7 3 0 5.8 7");
    define('w', 7, "0 7 1.75 0 3.5 5 5.25 0 7 7");
    define('x', 6, "0.4 7 5.6 0 ; 0.4 0 5.6 7");
    define('y', 6, "0.4 7 3 0 ; 5.6 7 2.2 -3");
    define('z', 6, "0.6 7 5.4 7 0.6 0 5.4 0");

    // ---- ponctuation et opérateurs
    define(' ', 4, "");
    define('+', 6, "3 1.6 3 7.6 ; 0 4.6 6 4.6");
    define('-', 6, "1 4.6 5 4.6");
    define('*', 6, "0.8 1.8 5.2 7.2 ; 0.8 7.2 5.2 1.8");
    define('/', 6, "0.6 -0.5 5.4 10.5");
    define('=', 6, "0.4 6.2 5.6 6.2 ; 0.4 3.0 5.6 3.0");
    define('(', 4, "a 5.2 5 3.6 6 125 235");
    define(')', 4, "a -1.2 5 3.6 6 55 -55");
    define('.', 3, "1.5 0.1 1.5 0.9");
    define(',', 3, "1.8 0.9 1.2 -1.4");
    define('!', 3, "1.5 10 1.5 3 ; 1.5 0.3 1.5 1.1");
    define('^', 6, "1 6.5 3 10 5 6.5");
    define('%', 6, "0.4 0 5.6 10 ; a 1.5 8.4 1.1 1.4 0 360 ; a 4.5 1.6 1.1 1.4 0 360");
    define(':', 3, "1.5 1 1.5 1.8 ; 1.5 5.4 1.5 6.2");
    define('\'', 3, "1.5 10 1.5 7.6");
    define('?', 6, "a 3 7.5 2.5 2.4 160 -70 3 3.6 3 2.8 ; 3 0.3 3 1.1");
    define('_', 6, "0 -1 6 -1");
    define('|', 3, "1.5 -1 1.5 11");

    // ---- symboles mathématiques (codes >= 0x80, voir Glyphs.h)
    define(glyph::kMinus, 6, "0.4 4.6 5.6 4.6");
    define(glyph::kTimes, 6, "1.0 2.6 5.0 6.6 ; 1.0 6.6 5.0 2.6");
    define(glyph::kDivide, 6, "0.4 4.6 5.6 4.6 ; 3 7.1 3 7.8 ; 3 1.4 3 2.1");
    define(glyph::kSqrt, 7.6f, "0 4.8 1.2 5.4 2.8 0 4.6 10.6 7.6 10.6");
    define(glyph::kPi, 6, "0.2 6.8 5.8 6.8 ; 1.8 6.8 1.6 0 ; 4.4 6.8 4.6 0");
    define(glyph::kPlusMinus, 6, "3 4 3 9 ; 0.4 6.5 5.6 6.5 ; 0.4 1.8 5.6 1.8");
    define(glyph::kBackspace, 7.5f, "2.2 8 7.5 8 7.5 2 2.2 2 0 5 2.2 8 ; 3.6 3.6 5.8 6.4 ; 3.6 6.4 5.8 3.6");
    define(glyph::kCheck, 6, "0.4 4.6 2.2 1.2 5.8 9");
    define(glyph::kDegree, 4, "a 2 8.4 1.4 1.4 0 360");
    define(glyph::kBullet, 3, "1.5 4.8 1.5 5.4");
    define(glyph::kArrowRight, 6, "0.2 5 5.6 5 ; 3.4 7.4 5.8 5 3.4 2.6");
}

TextLayout StrokeFont::layout(const std::string& text) const
{
    TextLayout out;
    float x = 0.0f;
    float lastSpacing = 0.0f;
    bool sup = false;
    for (unsigned char c : text) {
        if (c == static_cast<unsigned char>(glyph::kSupBegin)) { sup = true; continue; }
        if (c == static_cast<unsigned char>(glyph::kSupEnd)) { sup = false; continue; }
        const Glyph& g = glyphs_[c].defined ? glyphs_[c] : glyphs_[static_cast<unsigned char>('?')];
        const float s = sup ? kSupScale : 1.0f;
        const float raise = sup ? kSupRaise / kGrid : 0.0f;
        for (const auto& stroke : g.strokes) {
            Polyline2D pl;
            pl.scale = s;
            for (const float2& p : stroke) {
                const float2 q = make_float2(x + p.x * s, raise + p.y * s);
                if (!pl.points.empty()) {
                    const float dx = q.x - pl.points.back().x, dy = q.y - pl.points.back().y;
                    if (dx * dx + dy * dy < 1e-8f) continue;
                }
                pl.points.push_back(q);
            }
            if (pl.points.size() >= 2) out.strokes.push_back(std::move(pl));
        }
        lastSpacing = (kSpacing / kGrid) * s;
        x += g.advance * s + lastSpacing;
    }
    out.width = x > 0.0f ? x - lastSpacing : 0.0f;
    return out;
}

float StrokeFont::measure(const std::string& text) const
{
    float x = 0.0f;
    float lastSpacing = 0.0f;
    bool sup = false;
    for (unsigned char c : text) {
        if (c == static_cast<unsigned char>(glyph::kSupBegin)) { sup = true; continue; }
        if (c == static_cast<unsigned char>(glyph::kSupEnd)) { sup = false; continue; }
        const Glyph& g = glyphs_[c].defined ? glyphs_[c] : glyphs_[static_cast<unsigned char>('?')];
        const float s = sup ? kSupScale : 1.0f;
        lastSpacing = (kSpacing / kGrid) * s;
        x += g.advance * s + lastSpacing;
    }
    return x > 0.0f ? x - lastSpacing : 0.0f;
}

}  // namespace crtx
