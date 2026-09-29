// CalculatoRTX - génération procédurale de maillages triangulaires (accélérés par les RT cores).
#pragma once

#include "../common/VecMath.h"
#include "StrokeFont.h"

#include <functional>
#include <vector>

namespace crtx {

struct Mesh {
    std::vector<float3> positions;
    std::vector<float3> normals;  // vide => ombrage plat (normale géométrique)
    std::vector<float2> uvs;      // optionnel
    std::vector<uint3> indices;
    bool flat = false;

    bool empty() const { return indices.empty(); }
    void clear()
    {
        positions.clear();
        normals.clear();
        uvs.clear();
        indices.clear();
    }
};

// Pavé à coins arrondis (profil "lathe" autour d'un rectangle arrondi) avec congés
// supérieur/inférieur et creux optionnel (touches concaves).
struct RoundedBoxDesc {
    float3 center = make_float3(0, 0, 0);  // centre de la face inférieure
    float3 size = make_float3(1, 1, 1);    // (largeur X, hauteur Y, profondeur Z)
    float cornerRadius = 0.1f;             // rayon des coins verticaux
    float topFillet = 0.05f;               // congé de l'arête supérieure
    float bottomFillet = 0.0f;             // congé de l'arête inférieure
    float dish = 0.0f;                     // profondeur du creux de la face supérieure
    int cornerSegments = 7;
    int filletSegments = 5;
    int topRings = 7;
};

void addRoundedBox(Mesh& mesh, const RoundedBoxDesc& d);

// Hauteur de la face supérieure d'un RoundedBox en (x, z) locaux (pour poser du texte dessus).
float roundedBoxTopHeight(const RoundedBoxDesc& d, float x, float z);

// Quadrilatère subdivisé (sol, lumières, grille perforée).
void addGridQuad(Mesh& mesh, float3 corner, float3 edgeU, float3 edgeV, int nu, int nv, bool withNormals = true);

// Repère d'un texte posé sur une surface.
struct TextFrame {
    float3 origin;  // point (0,0) de la ligne de base
    float3 right;   // direction de lecture (unitaire)
    float3 up;      // haut des lettres (unitaire)
    float3 normal;  // normale de la surface (épaisseur des traits)
};

struct TextStyle {
    float size = 1.0f;          // hauteur de capitale (unités monde)
    float strokeWidth = 0.12f;  // largeur de trait relative à la taille
    float height = 0.02f;       // relief des traits (unités monde)
    float bevel = 0.22f;        // chanfrein du sommet des traits (relatif à la largeur)
};

enum class TextAlign { Left, Center, Right };

// Ajoute un texte extrudé (maillage à ombrage plat). baseHeight(x,z) permet de suivre
// une surface courbe (touche concave) ; retourne la largeur monde du texte.
float addText(Mesh& mesh, const StrokeFont& font, const std::string& text, const TextFrame& frame,
              const TextStyle& style, TextAlign align,
              const std::function<float(float3)>& baseOffset = nullptr);

}  // namespace crtx
