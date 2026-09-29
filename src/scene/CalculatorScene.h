// CalculatoRTX - description 3D complète de la calculatrice (géométrie, matériaux, lumières).
//
// Tout ce qui est visible à l'écran est de la géométrie triangulaire traversée par les
// rayons : châssis, touches, légendes en relief, chiffres de l'afficheur, vitre, grille
// perforée, cellule solaire, bureau et luminaires. Il n'y a aucun élément 2D.
#pragma once

#include "../render/LaunchParams.h"
#include "Keys.h"
#include "Mesh.h"
#include "StrokeFont.h"

#include <string>
#include <vector>

namespace crtx {

enum MaterialId : int {
    kMatDesk, kMatChassis, kMatKeybed, kMatBezel, kMatVfdPanel, kMatGlass, kMatGlow, kMatGrille,
    kMatGrilleFrame, kMatSolar, kMatBrand, kMatBrandGreen,
    kMatKeyLight, kMatKeyDark, kMatKeyOrange, kMatKeyGreen, kMatKeyRed, kMatKeyYellow,
    kMatLegendDark, kMatLegendLight, kMatLegendWhite,
    kMatVfdBright, kMatVfdDim, kMatVfdError, kMatVfdStatus,
    kMatLightKey, kMatLightRim, kMatLightFill,
    kMatCount
};

struct SceneMesh {
    Mesh mesh;
    bool alphaCutout = false;  // grille perforée : any-hit + Opacity Micromaps
    bool dirty = true;         // doit être (re)construit sur le GPU
};

struct SceneInstance {
    int mesh = -1;
    int material = 0;
    Affine transform = affineIdentity();
    unsigned mask = kMaskSolid;
    int pickId = -1;
    int lightIndex = -1;
    float glow = 0.0f;
    float3 glowColor = make_float3(0, 0, 0);
};

struct KeyDef {
    KeyId id;
    int row, col;
    std::string label;
    std::string altLabel;  // libellé en mode "2nd" (vide = identique)
    KeyStyle style;
};

struct DisplayContent {
    std::string expression;   // ligne 1 : saisie
    std::string main;         // ligne 2 : résultat / aperçu
    bool mainIsPreview = false;
    bool error = false;
    std::string statusLeft;   // "2nd  DEG  M"
    std::string statusRight;  // "GPU DD ✓"

    bool operator==(const DisplayContent& o) const
    {
        return expression == o.expression && main == o.main && mainIsPreview == o.mainIsPreview &&
               error == o.error && statusLeft == o.statusLeft && statusRight == o.statusRight;
    }
    bool operator!=(const DisplayContent& o) const { return !(*this == o); }
};

class CalculatorScene {
public:
    explicit CalculatorScene(const StrokeFont& font);

    void build();

    // ---- état interactif
    void setDisplay(const DisplayContent& content);
    void setShift(bool shift);
    void pressKey(int keyIndex);
    // Anime les touches ; retourne vrai si au moins une transformation a changé.
    bool animate(float dt, int hoveredKey);

    const std::vector<KeyDef>& keys() const { return keys_; }
    std::vector<SceneMesh>& meshes() { return meshes_; }
    const std::vector<SceneInstance>& instances() const { return instances_; }
    const std::vector<Material>& materials() const { return materials_; }
    const std::vector<RectLight>& lights() const { return lights_; }

    float3 focusPoint() const { return make_float3(0.0f, 1.2f, 0.75f); }

private:
    int addMesh(Mesh&& m, bool alpha = false);
    int addInstance(const SceneInstance& inst);
    void buildMaterials();
    void buildBody();
    void buildKeys();
    void buildLights();
    void rebuildLegend(int key);
    void rebuildDisplay();

    const StrokeFont& font_;
    std::vector<SceneMesh> meshes_;
    std::vector<SceneInstance> instances_;
    std::vector<Material> materials_;
    std::vector<RectLight> lights_;
    std::vector<KeyDef> keys_;

    struct KeyRuntime {
        int capInstance = -1;
        int legendInstance = -1;
        int legendMesh = -1;
        float press = 0.0f;  // 0..1 enfoncement
        float glow = 0.0f;
        float3 center;
    };
    std::vector<KeyRuntime> keyRt_;
    RoundedBoxDesc capDesc_;
    bool shift_ = false;

    DisplayContent display_;
    int displayMainMesh_ = -1, displayMainInst_ = -1;
    int displayExprMesh_ = -1, displayExprInst_ = -1;
    int displayStatusMesh_ = -1, displayStatusInst_ = -1;
    float3 panelCenter_;
    float2 panelHalf_;
    float panelTop_ = 0.0f;
};

}  // namespace crtx
