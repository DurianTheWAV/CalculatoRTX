// CalculatoRTX - police vectorielle "à traits" : chaque caractère est un ensemble de
// segments qui deviennent de vrais volumes 3D (triangles) traversés par les rayons.
// Aucune texture : le texte de l'afficheur et des touches est entièrement ray tracé.
#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace crtx {

struct Polyline2D {
    std::vector<float2> points;  // coordonnées en unités "hauteur de capitale = 1"
    float scale = 1.0f;          // 1 = taille normale, <1 pour les exposants
};

struct TextLayout {
    std::vector<Polyline2D> strokes;
    float width = 0.0f;  // largeur totale (mêmes unités)
};

class StrokeFont {
public:
    StrokeFont();

    // Mise en page d'une chaîne (octets, glyphes spéciaux >= 0x80, exposants entre {}).
    TextLayout layout(const std::string& text) const;

    // Largeur d'une chaîne sans générer les segments.
    float measure(const std::string& text) const;

private:
    struct Glyph {
        float advance = 0.6f;
        std::vector<std::vector<float2>> strokes;  // polylignes
        bool defined = false;
    };
    void define(unsigned char c, float advance, const char* program);
    Glyph glyphs_[256];
};

}  // namespace crtx
