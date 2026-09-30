// CalculatoRTX - construction de la scène 3D de la calculatrice.
#include "CalculatorScene.h"

#include "../common/Glyphs.h"
#include "../render/Grille.h"

#include <algorithm>
#include <cmath>

namespace crtx {

namespace {

// ---- dimensions (unités ~ centimètres)
constexpr float kBodyW = 21.8f, kBodyD = 15.3f, kBodyT = 1.2f;
constexpr float kKeySize = 1.55f, kKeyGap = 0.4f, kGroupGap = 0.9f, kKeyHeight = 0.55f;
constexpr float kKeybedT = 0.05f;
constexpr float kKeysZ0 = -0.85f;  // bord arrière de la première rangée
constexpr float kPressDepth = 0.15f;
constexpr float kDispBack = -kBodyD * 0.5f + 0.9f, kDispFront = kDispBack + 4.0f;  // cadre de l'afficheur
constexpr float kStripFront = kKeysZ0 - 0.35f;  // 0.35 = débord du lit des touches

// ---- coque transparente et électronique interne
constexpr float kWall = 0.14f;                 // épaisseur de la coque en plastique translucide
constexpr float kPcbY = 0.50f, kPcbT = 0.08f;  // circuit imprimé : face inférieure, épaisseur
constexpr float kPcbTop = kPcbY + kPcbT;

// ---- inclinaison : l'arrière se relève sur une béquille, l'avant repose sur deux patins
constexpr float kTilt = 15.0f * kPi / 180.0f;
constexpr float kStandAngle = 30.0f * kPi / 180.0f;  // béquille : angle avec la verticale
constexpr float kFootX = 8.5f, kFootZ = kBodyD * 0.5f - 0.7f, kFootR = 0.5f, kFootH = 0.12f, kFootFillet = 0.05f;

std::string G(unsigned char c) { return std::string(1, static_cast<char>(c)); }

Material mat(float3 base, float rough, float metallic = 0.0f)
{
    Material m{};
    m.baseColor = base;
    m.roughness = rough;
    m.metallic = metallic;
    m.emission = make_float3(0, 0, 0);
    m.transmission = 0.0f;
    m.ior = 1.5f;
    m.clearcoat = 0.0f;
    m.clearcoatRoughness = 0.05f;
    m.specular = 0.5f;
    m.pattern = kPatternNone;
    m.patternScale = 1.0f;
    m.absorbDistance = 0.0f;
    m.absorbColor = make_float3(1, 1, 1);
    m.haze = 0.0f;
    m.hazeColor = make_float3(0, 0, 0);
    m.emitUpOnly = 0.0f;
    return m;
}

Material emissive(float3 base, float3 emission, float rough = 0.5f)
{
    Material m = mat(base, rough);
    m.emission = emission;
    return m;
}

float keyX(int col)
{
    const float x0 = -(5 * kKeySize + 4 * kKeyGap) - kGroupGap * 0.5f;
    float x = x0 + kKeySize * 0.5f + col * (kKeySize + kKeyGap);
    if (col >= 5) x += kGroupGap;
    return x;
}

float keyZ(int row) { return kKeysZ0 + kKeySize * 0.5f + row * (kKeySize + kKeyGap); }

// Pavé (arêtes vives si les rayons sont nuls) et cylindre vertical, posés sur leur face inférieure
RoundedBoxDesc box(float3 bottomCenter, float3 size, float corner = 0.0f, float topFillet = 0.0f, float bottomFillet = 0.0f)
{
    RoundedBoxDesc d;
    d.center = bottomCenter;
    d.size = size;
    d.cornerRadius = corner;
    d.topFillet = topFillet;
    d.bottomFillet = bottomFillet;
    d.cornerSegments = corner > 0.0f ? 3 : 1;
    d.filletSegments = 2;
    d.topRings = 1;
    return d;
}

RoundedBoxDesc cylinder(float3 bottomCenter, float diameter, float height, float topFillet = 0.0f, int segments = 5)
{
    RoundedBoxDesc d = box(bottomCenter, make_float3(diameter, height, diameter), diameter * 0.5f, topFillet);
    d.cornerSegments = segments;
    return d;
}

}  // namespace

CalculatorScene::CalculatorScene(const StrokeFont& font) : font_(font) {}

int CalculatorScene::addMesh(Mesh&& m, bool alpha)
{
    SceneMesh sm;
    sm.mesh = std::move(m);
    sm.alphaCutout = alpha;
    sm.dirty = true;
    meshes_.push_back(std::move(sm));
    return static_cast<int>(meshes_.size() - 1);
}

int CalculatorScene::addInstance(const SceneInstance& inst)
{
    instances_.push_back(inst);
    return static_cast<int>(instances_.size() - 1);
}

int CalculatorScene::addPart(Mesh&& m, int material, unsigned mask, bool alpha)
{
    SceneInstance si;
    si.mesh = addMesh(std::move(m), alpha);
    si.material = material;
    si.mask = mask;
    si.transform = calcXf_;
    return addInstance(si);
}

void CalculatorScene::build()
{
    meshes_.clear();
    instances_.clear();
    lights_.clear();
    // Inclinaison autour de X ; la hauteur est choisie pour que le bord avant des patins
    // (cercle du congé inférieur) touche juste le bureau.
    const float c = std::cos(kTilt), s = std::sin(kTilt);
    const float lift = kFootFillet - 0.002f + c * (kFootH - kFootFillet) + s * (kFootZ + kFootR - kFootFillet);
    calcXf_ = affineMul(affineTranslate(make_float3(0.0f, lift, 0.0f)), affineRotateX(kTilt));
    focus_ = xformPoint(calcXf_.m, make_float3(0.0f, kBodyT, 0.3f));
    buildMaterials();
    buildLights();
    buildBody();
    buildInternals();
    buildStand();
    buildKeys();
    rebuildDisplay();
}

void CalculatorScene::buildMaterials()
{
    materials_.assign(kMatCount, mat(make_float3(0.5f, 0.5f, 0.5f), 0.5f));
    const float3 nvGreen = make_float3(0.18f, 0.55f, 0.0f);
    const float3 vfd = make_float3(0.25f, 1.0f, 0.82f);

    Material desk = mat(make_float3(0.20f, 0.105f, 0.05f), 0.38f);
    desk.clearcoat = 1.0f;
    desk.clearcoatRoughness = 0.04f;
    desk.pattern = kPatternWood;
    desk.patternScale = 0.9f;
    materials_[kMatDesk] = desk;

    // Coque : polycarbonate gris translucide, légèrement dépoli. La diffusion (haze) garde
    // la teinte grise, l'absorption dans l'épaisseur laisse seulement deviner l'électronique.
    Material shell = mat(make_float3(0.97f, 0.975f, 0.98f), 0.15f);
    shell.transmission = 1.0f;
    shell.ior = 1.58f;
    shell.absorbColor = make_float3(0.45f, 0.46f, 0.48f);
    shell.absorbDistance = 0.30f;
    shell.haze = 0.22f;
    shell.hazeColor = make_float3(0.50f, 0.51f, 0.53f);
    materials_[kMatShell] = shell;

    Material keybed = mat(make_float3(0.025f, 0.026f, 0.03f), 0.35f);
    keybed.pattern = kPatternCarbon;
    keybed.patternScale = 7.0f;
    keybed.clearcoat = 0.7f;
    materials_[kMatKeybed] = keybed;

    materials_[kMatBezel] = mat(make_float3(0.008f, 0.008f, 0.01f), 0.07f);

    Material panel = mat(make_float3(0.008f, 0.02f, 0.02f), 0.55f);
    panel.pattern = kPatternVfd;
    panel.patternScale = 18.0f;
    materials_[kMatVfdPanel] = panel;

    Material glass = mat(make_float3(0.96f, 0.995f, 0.985f), 0.0f);
    glass.transmission = 1.0f;
    glass.ior = 1.5f;
    materials_[kMatGlass] = glass;

    materials_[kMatGlow] = emissive(make_float3(0.1f, 0.3f, 0.0f), nvGreen * 7.0f);

    Material grille = mat(make_float3(0.16f, 0.165f, 0.17f), 0.33f, 1.0f);
    grille.pattern = kPatternGrille;
    materials_[kMatGrille] = grille;
    materials_[kMatGrilleFrame] = mat(make_float3(0.12f, 0.12f, 0.13f), 0.25f, 1.0f);

    Material solar = mat(make_float3(0.035f, 0.02f, 0.045f), 0.12f);
    solar.pattern = kPatternSolar;
    solar.patternScale = 4.0f;
    solar.clearcoat = 1.0f;
    materials_[kMatSolar] = solar;

    // Inscriptions du bandeau transparent : la mention verte lumineuse n'éclaire que vers le
    // haut (sinon son reflet réapparaîtrait sous le plastique).
    materials_[kMatBrand] = mat(make_float3(0.92f, 0.92f, 0.95f), 0.42f, 1.0f);  // argent satiné
    materials_[kMatBrandGreen] = emissive(make_float3(0.1f, 0.3f, 0.0f), nvGreen * 4.0f, 0.4f);
    materials_[kMatBrandGreen].emitUpOnly = 1.0f;

    auto keyMat = [](float3 c) {
        Material m = mat(c, 0.32f);
        m.clearcoat = 0.35f;
        m.clearcoatRoughness = 0.12f;
        return m;
    };
    materials_[kMatKeyLight] = keyMat(make_float3(0.60f, 0.59f, 0.56f));
    materials_[kMatKeyDark] = keyMat(make_float3(0.032f, 0.034f, 0.038f));
    materials_[kMatKeyOrange] = keyMat(make_float3(0.90f, 0.27f, 0.025f));
    materials_[kMatKeyGreen] = keyMat(make_float3(0.20f, 0.52f, 0.0f));
    materials_[kMatKeyRed] = keyMat(make_float3(0.62f, 0.035f, 0.03f));
    materials_[kMatKeyYellow] = keyMat(make_float3(0.85f, 0.58f, 0.04f));

    materials_[kMatLegendDark] = mat(make_float3(0.015f, 0.015f, 0.018f), 0.45f);
    materials_[kMatLegendLight] = emissive(make_float3(0.8f, 0.85f, 0.9f), make_float3(0.35f, 0.55f, 0.6f), 0.4f);
    materials_[kMatLegendWhite] = emissive(make_float3(0.95f, 0.95f, 0.95f), make_float3(0.3f, 0.3f, 0.3f), 0.4f);

    materials_[kMatVfdBright] = emissive(make_float3(0.1f, 0.4f, 0.35f), vfd * 5.0f, 0.3f);
    materials_[kMatVfdDim] = emissive(make_float3(0.05f, 0.2f, 0.18f), vfd * 1.5f, 0.3f);
    materials_[kMatVfdError] = emissive(make_float3(0.4f, 0.1f, 0.05f), make_float3(1.0f, 0.22f, 0.08f) * 5.0f, 0.3f);
    materials_[kMatVfdStatus] = emissive(make_float3(0.1f, 0.3f, 0.3f), vfd * 2.2f, 0.3f);

    // ---- électronique interne
    Material pcb = mat(make_float3(0.012f, 0.075f, 0.032f), 0.4f);
    pcb.clearcoat = 0.8f;
    pcb.clearcoatRoughness = 0.12f;
    pcb.pattern = kPatternPcb;
    materials_[kMatPcb] = pcb;
    materials_[kMatSteel] = mat(make_float3(0.78f, 0.78f, 0.80f), 0.22f, 1.0f);
    materials_[kMatTin] = mat(make_float3(0.72f, 0.72f, 0.74f), 0.35f, 1.0f);
    materials_[kMatBrass] = mat(make_float3(0.85f, 0.62f, 0.30f), 0.30f, 1.0f);
    Material chip = mat(make_float3(0.018f, 0.018f, 0.02f), 0.5f);
    chip.clearcoat = 0.25f;
    materials_[kMatChip] = chip;
    materials_[kMatDie] = mat(make_float3(0.30f, 0.32f, 0.42f), 0.06f, 1.0f);  // silicium poli
    materials_[kMatCeramic] = mat(make_float3(0.42f, 0.31f, 0.18f), 0.45f);
    Material sleeve = mat(make_float3(0.02f, 0.04f, 0.20f), 0.3f);
    sleeve.clearcoat = 0.8f;
    materials_[kMatCapSleeve] = sleeve;
    materials_[kMatAluminium] = mat(make_float3(0.88f, 0.89f, 0.91f), 0.18f, 1.0f);
    Material kapton = mat(make_float3(0.55f, 0.28f, 0.03f), 0.3f);
    kapton.clearcoat = 0.7f;
    materials_[kMatKapton] = kapton;
    materials_[kMatFerrite] = mat(make_float3(0.05f, 0.05f, 0.055f), 0.55f);
    materials_[kMatLed] = emissive(make_float3(0.2f, 0.6f, 0.1f), make_float3(0.25f, 1.0f, 0.08f) * 5.0f, 0.3f);
    materials_[kMatRubber] = mat(make_float3(0.015f, 0.015f, 0.016f), 0.85f);
    Material stand = mat(make_float3(0.34f, 0.35f, 0.37f), 0.30f, 1.0f);
    stand.pattern = kPatternBrushed;
    materials_[kMatStand] = stand;

    materials_[kMatLightKey] = emissive(make_float3(0, 0, 0), make_float3(1.0f, 0.93f, 0.85f) * 9.0f);
    materials_[kMatLightRim] = emissive(make_float3(0, 0, 0), make_float3(0.65f, 0.8f, 1.0f) * 7.0f);
    materials_[kMatLightFill] = emissive(make_float3(0, 0, 0), make_float3(1.0f, 0.75f, 0.55f) * 3.0f);
}

void CalculatorScene::buildLights()
{
    const float3 target = make_float3(0.0f, 2.2f, 0.0f);
    struct L {
        float3 center;
        float w, h;
        int material;
    };
    const L defs[] = {
        {make_float3(-11.0f, 25.0f, 13.0f), 15.0f, 9.0f, kMatLightKey},
        {make_float3(3.0f, 11.0f, -21.0f), 24.0f, 2.6f, kMatLightRim},
        {make_float3(19.0f, 9.0f, 9.0f), 5.0f, 7.0f, kMatLightFill},
    };
    for (const L& d : defs) {
        const float3 n = normalize(target - d.center);  // orientée vers la calculatrice
        float3 u = normalize(cross(make_float3(0, 1, 0), n));
        if (length(u) < 0.5f) u = make_float3(1, 0, 0);
        const float3 v = normalize(cross(n, u));
        RectLight rl{};
        rl.edgeU = u * d.w;
        rl.edgeV = v * d.h;
        rl.corner = d.center - rl.edgeU * 0.5f - rl.edgeV * 0.5f;
        rl.normal = normalize(cross(rl.edgeU, rl.edgeV));
        if (dot(rl.normal, n) < 0.0f) {  // garantit une émission vers la scène
            rl.edgeU = rl.edgeU * -1.0f;
            rl.corner = d.center - rl.edgeU * 0.5f - rl.edgeV * 0.5f;
            rl.normal = normalize(cross(rl.edgeU, rl.edgeV));
        }
        rl.emission = materials_[d.material].emission;
        rl.area = d.w * d.h;
        lights_.push_back(rl);

        Mesh m;
        addGridQuad(m, rl.corner, rl.edgeU, rl.edgeV, 1, 1);
        SceneInstance si;
        si.mesh = addMesh(std::move(m));
        si.material = d.material;
        si.mask = kMaskLight;
        si.lightIndex = static_cast<int>(lights_.size() - 1);
        addInstance(si);
    }
}

void CalculatorScene::buildBody()
{
    // ---- bureau (repère monde)
    {
        Mesh m;
        addGridQuad(m, make_float3(-70, 0, 70), make_float3(140, 0, 0), make_float3(0, 0, -140), 1, 1);
        SceneInstance si;
        si.mesh = addMesh(std::move(m));
        si.material = kMatDesk;
        addInstance(si);
    }
    auto addBox = [&](const RoundedBoxDesc& d, int material, unsigned mask = kMaskSolid) {
        Mesh m;
        addRoundedBox(m, d);
        return addPart(std::move(m), material, mask);
    };
    // ---- coque creuse en plastique translucide : paroi extérieure + paroi intérieure (normales
    //      retournées vers la cavité), soit un volume d'épaisseur kWall que les rayons
    //      traversent en réfractant deux fois. Masque "verre" : ne bloque pas les ombres.
    RoundedBoxDesc body;
    body.center = make_float3(0, 0, 0);
    body.size = make_float3(kBodyW, kBodyT, kBodyD);
    body.cornerRadius = 1.3f;
    body.topFillet = 0.38f;
    body.bottomFillet = 0.22f;
    body.cornerSegments = 10;
    body.filletSegments = 7;
    addBox(body, kMatShell, kMaskGlass);
    {
        RoundedBoxDesc cavity = body;  // décalage exact de la surface extérieure
        cavity.center = make_float3(0, kWall, 0);
        cavity.size = body.size - make_float3(2 * kWall, 2 * kWall, 2 * kWall);
        cavity.cornerRadius = body.cornerRadius - kWall;
        cavity.topFillet = body.topFillet - kWall;
        cavity.bottomFillet = std::max(body.bottomFillet - kWall, 0.02f);
        Mesh m;
        addRoundedBox(m, cavity);
        invertMesh(m);
        addPart(std::move(m), kMatShell, kMaskGlass);
    }

    // Les plaques opaques (lit des touches, cadre de l'afficheur, cellule solaire) sont
    // enchâssées à mi-épaisseur dans le dessus de la coque : aucune face coplanaire.
    const float inset = kWall * 0.5f;

    // ---- lit des touches (fibre de carbone)
    const float keysBack = kKeysZ0, keysFront = keyZ(3) + kKeySize * 0.5f;
    RoundedBoxDesc bed;
    bed.center = make_float3(0, kBodyT - inset, 0.5f * (keysBack + keysFront));
    bed.size = make_float3(-2.0f * keyX(0) + kKeySize + 0.7f, kKeybedT + inset, keysFront - keysBack + 0.7f);
    bed.cornerRadius = 0.6f;
    bed.topFillet = 0.02f;
    bed.topRings = 2;
    addBox(bed, kMatKeybed);

    // ---- cadre de l'afficheur
    RoundedBoxDesc bezel;
    bezel.center = make_float3(0, kBodyT - inset, 0.5f * (kDispBack + kDispFront));
    bezel.size = make_float3(kBodyW - 2.0f, 0.14f + inset, kDispFront - kDispBack);
    bezel.cornerRadius = 0.5f;
    bezel.topFillet = 0.06f;
    bezel.topRings = 2;
    addBox(bezel, kMatBezel);
    const float bezelTop = kBodyT + 0.14f;

    // ---- fond de l'afficheur VFD + vitre
    panelCenter_ = make_float3(1.7f, bezelTop, bezel.center.z);
    panelHalf_ = make_float2(7.7f, 1.65f);
    RoundedBoxDesc panel;
    panel.center = panelCenter_;
    panel.size = make_float3(2 * panelHalf_.x, 0.02f, 2 * panelHalf_.y);
    panel.cornerRadius = 0.2f;
    panel.topFillet = 0.0f;
    panel.topRings = 2;
    addBox(panel, kMatVfdPanel);
    panelTop_ = bezelTop + 0.02f;

    RoundedBoxDesc glass;
    glass.center = make_float3(panelCenter_.x, panelTop_ + 0.045f, panelCenter_.z);
    glass.size = make_float3(2 * panelHalf_.x + 0.4f, 0.13f, 2 * panelHalf_.y + 0.4f);
    glass.cornerRadius = 0.3f;
    glass.topFillet = 0.05f;
    glass.bottomFillet = 0.02f;
    glass.topRings = 2;
    addBox(glass, kMatGlass, kMaskGlass);

    // ---- grille perforée (Opacity Micromaps) au-dessus d'une bande lumineuse verte
    const float gx1 = panelCenter_.x - panelHalf_.x - 0.45f, gx0 = gx1 - grille::kWidth;
    const float gz0 = panelCenter_.z - grille::kHeight * 0.5f, gz1 = panelCenter_.z + grille::kHeight * 0.5f;
    RoundedBoxDesc glow;
    glow.center = make_float3(0.5f * (gx0 + gx1), bezelTop, 0.5f * (gz0 + gz1));
    glow.size = make_float3(gx1 - gx0 - 0.1f, 0.01f, gz1 - gz0 - 0.1f);
    glow.cornerRadius = 0.1f;
    glow.topFillet = 0.0f;
    glow.topRings = 1;
    addBox(glow, kMatGlow, kMaskSolid);
    {
        Mesh m;
        const float gy = bezelTop + 0.07f;
        addGridQuad(m, make_float3(gx0, gy, gz1), make_float3(gx1 - gx0, 0, 0), make_float3(0, 0, gz0 - gz1), 16, 16);
        addPart(std::move(m), kMatGrille, kMaskSolid, true);
        // cadre de la grille
        const float fw = 0.12f;
        const RoundedBoxDesc bars[4] = {
            {make_float3(0.5f * (gx0 + gx1), bezelTop, gz0), make_float3(gx1 - gx0 + fw, 0.09f, fw), 0.05f, 0.03f, 0.0f, 0.0f, 3, 2, 1},
            {make_float3(0.5f * (gx0 + gx1), bezelTop, gz1), make_float3(gx1 - gx0 + fw, 0.09f, fw), 0.05f, 0.03f, 0.0f, 0.0f, 3, 2, 1},
            {make_float3(gx0, bezelTop, 0.5f * (gz0 + gz1)), make_float3(fw, 0.09f, gz1 - gz0 + fw), 0.05f, 0.03f, 0.0f, 0.0f, 3, 2, 1},
            {make_float3(gx1, bezelTop, 0.5f * (gz0 + gz1)), make_float3(fw, 0.09f, gz1 - gz0 + fw), 0.05f, 0.03f, 0.0f, 0.0f, 3, 2, 1},
        };
        for (const auto& b : bars) addBox(b, kMatGrilleFrame);
    }

    // ---- bandeau transparent entre l'afficheur et les touches : marque, mentions,
    //      cellule solaire (étroite, alignée à droite)
    const float stripZ = 0.5f * (kDispFront + kStripFront);
    const float left = keyX(0) - kKeySize * 0.5f, right = keyX(9) + kKeySize * 0.5f;
    RoundedBoxDesc solar;
    solar.size = make_float3(3.1f, 0.035f + 0.03f, 0.62f);
    solar.center = make_float3(right - solar.size.x * 0.5f, kBodyT - 0.03f, stripZ);
    solar.cornerRadius = 0.08f;
    solar.topFillet = 0.01f;
    solar.topRings = 1;
    addBox(solar, kMatSolar);

    const TextFrame brandFrame{make_float3(left, kBodyT, stripZ + 0.25f), make_float3(1, 0, 0), make_float3(0, 0, -1),
                               make_float3(0, 1, 0)};
    float brandEnd = left;
    {
        Mesh m;
        TextStyle st;
        st.size = 0.5f;
        st.strokeWidth = 0.15f;
        st.height = 0.025f;
        brandEnd += addText(m, font_, "CalculatoRTX", brandFrame, st, TextAlign::Left);
        addPart(std::move(m), kMatBrand);
    }
    {
        Mesh m;
        TextStyle st;
        st.size = 0.24f;
        st.strokeWidth = 0.16f;
        st.height = 0.015f;
        TextFrame f = brandFrame;
        const float solarLeft = solar.center.x - solar.size.x * 0.5f;
        f.origin = make_float3(0.5f * (brandEnd + solarLeft), kBodyT, stripZ + 0.12f);
        // mention du GPU et des technologies utilisées ('|' = puce séparatrice), réduite si trop longue
        std::string text;
        for (char c : badge_) text += c == '|' ? std::string(" ") + G(glyph::kBullet) + " " : std::string(1, c);
        const float room = solarLeft - brandEnd - 0.8f;
        const float w = font_.measure(text) * st.size;
        if (w > room) st.size *= room / w;
        addText(m, font_, text, f, st, TextAlign::Center);
        addPart(std::move(m), kMatBrandGreen);
    }
}

void CalculatorScene::buildInternals()
{
    // Électronique enfermée dans la cavité : x ∈ ±(kBodyW/2 - kWall), z ∈ ±(kBodyD/2 - kWall),
    // y ∈ [kWall, kBodyT - kWall]. Les composants sont surtout placés sous le pourtour
    // transparent (bords, bandeau) : c'est là qu'on les devine à travers le plastique.
    // Un maillage par matériau (un seul BLAS par famille de composants).
    std::vector<Mesh> parts(kMatCount);
    auto add = [&](int material, const RoundedBoxDesc& d) { addRoundedBox(parts[material], d); };
    const float cx = kBodyW * 0.5f - kWall, cz = kBodyD * 0.5f - kWall;
    const float zs = 0.5f * (kDispFront + kStripFront);  // milieu du bandeau
    const float margin = 0.175f;

    // ---- circuit imprimé sur entretoises en laiton, vissé
    RoundedBoxDesc pcb = box(make_float3(0, kPcbY, 0), make_float3(2 * (cx - margin), kPcbT, 2 * (cz - margin)),
                             1.3f - kWall - margin, 0.015f, 0.015f);
    pcb.cornerSegments = 8;
    add(kMatPcb, pcb);
    const float2 posts[] = {{-10.1f, -6.85f}, {10.1f, -6.85f}, {-10.1f, 6.85f},
                            {10.1f, 6.85f},   {-10.1f, zs},    {10.1f, zs}};
    for (const float2 p : posts) {
        add(kMatBrass, cylinder(make_float3(p.x, kWall * 0.5f, p.y), 0.34f, kPcbY - kWall * 0.5f + 0.01f));
        add(kMatSteel, cylinder(make_float3(p.x, kPcbTop - 0.01f, p.y), 0.38f, 0.075f, 0.035f));
    }

    // ---- dômes métalliques des contacts, sous chaque touche
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 10; ++col)
            add(kMatSteel, cylinder(make_float3(keyX(col), kPcbTop - 0.005f, keyZ(row)), 0.95f, 0.065f, 0.055f));

    // ---- sous le bandeau : processeur (substrat + silicium poli), quartz, mémoire, régulateur
    add(kMatChip, box(make_float3(-1.3f, kPcbTop, zs), make_float3(1.3f, 0.07f, 1.3f), 0.05f, 0.01f));
    add(kMatDie, box(make_float3(-1.3f, kPcbTop + 0.07f, zs), make_float3(0.72f, 0.05f, 0.72f), 0.02f, 0.008f));
    add(kMatAluminium, box(make_float3(1.25f, kPcbTop + 0.01f, zs), make_float3(0.95f, 0.3f, 0.4f), 0.12f, 0.08f));
    for (float sx : {-1.0f, 1.0f})
        add(kMatTin, box(make_float3(1.25f + sx * 0.42f, kPcbTop, zs), make_float3(0.22f, 0.03f, 0.3f)));
    add(kMatChip, box(make_float3(3.4f, kPcbTop + 0.02f, zs), make_float3(1.5f, 0.1f, 0.72f), 0.02f, 0.01f));
    for (int i = 0; i < 12; ++i)
        for (float sz : {-1.0f, 1.0f})
            add(kMatTin, box(make_float3(3.4f - 0.605f + i * 0.11f, kPcbTop, zs + sz * 0.41f), make_float3(0.05f, 0.035f, 0.16f)));
    add(kMatChip, box(make_float3(-6.0f, kPcbTop + 0.02f, zs), make_float3(0.3f, 0.11f, 0.16f)));
    for (float px : {-0.1f, 0.1f})
        add(kMatTin, box(make_float3(-6.0f + px, kPcbTop, zs + 0.12f), make_float3(0.04f, 0.04f, 0.1f)));
    add(kMatTin, box(make_float3(-6.0f, kPcbTop, zs - 0.12f), make_float3(0.04f, 0.04f, 0.1f)));

    // ---- composants CMS (condensateurs céramique, résistances) le long des bords
    unsigned seed = 0x4070u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(seed >> 8) * (1.0f / 16777216.0f);
    };
    auto passive = [&](float x, float z, bool alongX) {
        const bool resistor = rnd() < 0.4f;
        const float len = 0.32f, wid = 0.16f, hgt = resistor ? 0.05f : 0.1f;
        const float3 bodySize = alongX ? make_float3(len - 0.13f, hgt, wid) : make_float3(wid, hgt, len - 0.13f);
        add(resistor ? kMatChip : kMatCeramic, box(make_float3(x, kPcbTop, z), bodySize));
        for (float s : {-1.0f, 1.0f}) {
            const float o = s * (len * 0.5f - 0.035f);
            const float3 c = alongX ? make_float3(x + o, kPcbTop, z) : make_float3(x, kPcbTop, z + o);
            const float3 sz = alongX ? make_float3(0.07f, hgt + 0.005f, wid + 0.005f) : make_float3(wid + 0.005f, hgt + 0.005f, 0.07f);
            add(kMatTin, box(c, sz));
        }
    };
    for (float sx : {-1.0f, 1.0f})
        for (float z = -6.3f; z <= 6.31f; z += 0.62f)
            if (std::fabs(z - zs) > 0.35f && rnd() > 0.3f) passive(sx * 10.38f, z, false);
    for (float x = -9.4f; x <= 9.41f; x += 0.7f) {
        if (std::fabs(x - 9.05f) > 0.5f && std::fabs(x + 9.1f) > 0.5f && rnd() > 0.25f) passive(x, 7.12f, true);
        const bool nearCap = std::fabs(std::fabs(x) - 9.2f) < 0.55f, nearFlex = x > -4.8f && x < -1.2f;
        if (!nearCap && !nearFlex && rnd() > 0.25f) passive(x, -7.08f, true);
    }
    for (float x : {-8.8f, -8.2f, -7.6f, -7.0f, -2.35f, -0.25f, 0.45f, 2.15f, 4.6f})
        for (float sz : {-1.0f, 1.0f}) passive(x, zs + sz * 0.35f, false);

    // ---- condensateurs électrolytiques (angles arrière)
    for (float sx : {-1.0f, 1.0f}) {
        const float3 c = make_float3(sx * 9.2f, kPcbTop, -6.95f);
        add(kMatChip, box(c, make_float3(0.56f, 0.05f, 0.56f), 0.04f));
        add(kMatCapSleeve, cylinder(c + make_float3(0, 0.04f, 0), 0.52f, 0.36f, 0.04f, 6));
        add(kMatAluminium, cylinder(c + make_float3(0, 0.39f, 0), 0.44f, 0.02f, 0.01f, 6));
    }

    // ---- nappe souple (Kapton) vers l'afficheur, avec son connecteur
    add(kMatChip, box(make_float3(-3.0f, kPcbTop, -5.75f), make_float3(3.3f, 0.14f, 0.4f), 0.03f, 0.01f));
    add(kMatKapton, box(make_float3(-3.0f, kPcbTop, -6.6f), make_float3(3.0f, 0.02f, 1.3f)));
    add(kMatKapton, box(make_float3(-3.0f, kPcbTop, -7.26f), make_float3(3.0f, 1.02f - kPcbTop, 0.02f)));
    add(kMatKapton, box(make_float3(-3.0f, 1.0f, -6.6f), make_float3(3.0f, 0.02f, 1.3f)));

    // ---- avant : bobine d'alimentation, LED témoin verte ; dessous : pile bouton
    add(kMatFerrite, box(make_float3(-9.1f, kPcbTop, 7.02f), make_float3(0.5f, 0.3f, 0.5f), 0.06f, 0.04f));
    add(kMatLed, box(make_float3(9.05f, kPcbTop, 7.12f), make_float3(0.22f, 0.09f, 0.14f), 0.02f, 0.02f));
    RoundedBoxDesc cell = cylinder(make_float3(-7.8f, kPcbY - 0.32f, 5.6f), 2.0f, 0.32f, 0.04f, 10);
    cell.bottomFillet = 0.04f;
    add(kMatSteel, cell);

    for (int mtl = 0; mtl < kMatCount; ++mtl)
        if (!parts[mtl].empty()) addPart(std::move(parts[mtl]), mtl);
}

void CalculatorScene::buildStand()
{
    // ---- patins avant en caoutchouc, noyés à mi-épaisseur dans le fond de la coque
    for (float sx : {-1.0f, 1.0f}) {
        RoundedBoxDesc f = cylinder(make_float3(sx * kFootX, -kFootH, kFootZ), 2.0f * kFootR, kFootH + kWall * 0.5f, 0.0f, 8);
        f.bottomFillet = kFootFillet;
        Mesh m;
        addRoundedBox(m, f);
        addPart(std::move(m), kMatRubber);
    }
    // ---- charnière (repère calculatrice) et béquille en aluminium brossé (repère monde)
    const float zh = -kBodyD * 0.5f + 1.6f;
    {
        Mesh m;
        addRoundedBox(m, box(make_float3(0, -0.24f, zh), make_float3(8.8f, 0.24f + kWall * 0.5f, 0.7f), 0.25f, 0.0f, 0.08f));
        addPart(std::move(m), kMatStand);
    }
    const float3 hinge = xformPoint(calcXf_.m, make_float3(0.0f, -0.12f, zh));
    const float foot = -0.02f;  // légèrement enfoncée dans le bureau
    const float rise = hinge.y - foot;
    const float len = rise / std::cos(kStandAngle) + 0.06f;
    const float3 contact = make_float3(0.0f, foot, hinge.z - rise * std::tan(kStandAngle));
    Mesh m;
    addRoundedBox(m, box(make_float3(0, 0, 0), make_float3(8.0f, len, 0.22f), 0.11f, 0.08f, 0.08f));
    SceneInstance si;
    si.mesh = addMesh(std::move(m));
    si.material = kMatStand;
    si.transform = affineMul(affineTranslate(contact), affineRotateX(kStandAngle));
    addInstance(si);
}

void CalculatorScene::buildKeys()
{
    const std::string m1 = std::string("{") + G(glyph::kMinus) + "1}";
    keys_ = {
        {KeyId::Second, 0, 0, "2nd", "", KeyStyle::Shift},
        {KeyId::Sin, 0, 1, "sin", "sin" + m1, KeyStyle::Function},
        {KeyId::Cos, 0, 2, "cos", "cos" + m1, KeyStyle::Function},
        {KeyId::Tan, 0, 3, "tan", "tan" + m1, KeyStyle::Function},
        {KeyId::Drg, 0, 4, "DRG", "", KeyStyle::Function},
        {KeyId::D7, 0, 5, "7", "", KeyStyle::Number},
        {KeyId::D8, 0, 6, "8", "", KeyStyle::Number},
        {KeyId::D9, 0, 7, "9", "", KeyStyle::Number},
        {KeyId::Div, 0, 8, G(glyph::kDivide), "", KeyStyle::Operator},
        {KeyId::AllClear, 0, 9, "AC", "", KeyStyle::Clear},

        {KeyId::Square, 1, 0, "x{2}", "x{3}", KeyStyle::Function},
        {KeyId::Pow, 1, 1, "x{y}", "{y}" + G(glyph::kSqrt) + "x", KeyStyle::Function},
        {KeyId::Sqrt, 1, 2, G(glyph::kSqrt), "{3}" + G(glyph::kSqrt), KeyStyle::Function},
        {KeyId::Ln, 1, 3, "ln", "e{x}", KeyStyle::Function},
        {KeyId::Log, 1, 4, "log", "10{x}", KeyStyle::Function},
        {KeyId::D4, 1, 5, "4", "", KeyStyle::Number},
        {KeyId::D5, 1, 6, "5", "", KeyStyle::Number},
        {KeyId::D6, 1, 7, "6", "", KeyStyle::Number},
        {KeyId::Mul, 1, 8, G(glyph::kTimes), "", KeyStyle::Operator},
        {KeyId::Backspace, 1, 9, G(glyph::kBackspace), "", KeyStyle::Function},

        {KeyId::LParen, 2, 0, "(", "", KeyStyle::Function},
        {KeyId::RParen, 2, 1, ")", "", KeyStyle::Function},
        {KeyId::Pi, 2, 2, G(glyph::kPi), "", KeyStyle::Function},
        {KeyId::Euler, 2, 3, "e", "", KeyStyle::Function},
        {KeyId::Fact, 2, 4, "n!", "", KeyStyle::Function},
        {KeyId::D1, 2, 5, "1", "", KeyStyle::Number},
        {KeyId::D2, 2, 6, "2", "", KeyStyle::Number},
        {KeyId::D3, 2, 7, "3", "", KeyStyle::Number},
        {KeyId::Sub, 2, 8, G(glyph::kMinus), "", KeyStyle::Operator},
        {KeyId::Ans, 2, 9, "ANS", "", KeyStyle::Function},

        {KeyId::MemClear, 3, 0, "MC", "", KeyStyle::Function},
        {KeyId::MemRecall, 3, 1, "MR", "", KeyStyle::Function},
        {KeyId::MemPlus, 3, 2, "M+", "M" + G(glyph::kMinus), KeyStyle::Function},
        {KeyId::Inverse, 3, 3, "x" + m1, "", KeyStyle::Function},
        {KeyId::Exp, 3, 4, "EXP", "", KeyStyle::Function},
        {KeyId::D0, 3, 5, "0", "", KeyStyle::Number},
        {KeyId::Dot, 3, 6, ".", "", KeyStyle::Number},
        {KeyId::Negate, 3, 7, G(glyph::kPlusMinus), "", KeyStyle::Number},
        {KeyId::Add, 3, 8, "+", "", KeyStyle::Operator},
        {KeyId::Equals, 3, 9, "=", "", KeyStyle::Equals},
    };

    capDesc_ = RoundedBoxDesc{};
    capDesc_.size = make_float3(kKeySize, kKeyHeight, kKeySize);
    capDesc_.cornerRadius = 0.34f;
    capDesc_.topFillet = 0.17f;
    capDesc_.bottomFillet = 0.03f;
    capDesc_.dish = 0.04f;
    capDesc_.cornerSegments = 8;
    capDesc_.filletSegments = 6;
    capDesc_.topRings = 8;

    // Un seul maillage de touche, instancié 40 fois (instancing matériel de l'IAS)
    Mesh cap;
    RoundedBoxDesc local = capDesc_;
    local.center = make_float3(0, 0, 0);
    addRoundedBox(cap, local);
    const int capMesh = addMesh(std::move(cap));

    keyRt_.assign(keys_.size(), KeyRuntime{});
    for (size_t i = 0; i < keys_.size(); ++i) {
        const KeyDef& k = keys_[i];
        KeyRuntime& rt = keyRt_[i];
        rt.center = make_float3(keyX(k.col), kBodyT + kKeybedT, keyZ(k.row));
        int capMat = kMatKeyDark;
        switch (k.style) {
            case KeyStyle::Number: capMat = kMatKeyLight; break;
            case KeyStyle::Function: capMat = kMatKeyDark; break;
            case KeyStyle::Operator: capMat = kMatKeyOrange; break;
            case KeyStyle::Equals: capMat = kMatKeyGreen; break;
            case KeyStyle::Clear: capMat = kMatKeyRed; break;
            case KeyStyle::Shift: capMat = kMatKeyYellow; break;
        }
        SceneInstance si;
        si.mesh = capMesh;
        si.material = capMat;
        si.transform = affineMul(calcXf_, affineTranslate(rt.center));
        si.pickId = static_cast<int>(i);
        rt.capInstance = addInstance(si);

        rt.legendMesh = addMesh(Mesh{});
        SceneInstance li;
        li.mesh = rt.legendMesh;
        li.material = (k.style == KeyStyle::Number || k.style == KeyStyle::Shift) ? kMatLegendDark
                    : (k.style == KeyStyle::Function) ? kMatLegendLight : kMatLegendWhite;
        li.transform = si.transform;
        li.pickId = static_cast<int>(i);
        rt.legendInstance = addInstance(li);
        rebuildLegend(static_cast<int>(i));
    }
}

void CalculatorScene::rebuildLegend(int key)
{
    const KeyDef& k = keys_[key];
    KeyRuntime& rt = keyRt_[key];
    const std::string& text = (shift_ && !k.altLabel.empty()) ? k.altLabel : k.label;
    Mesh& m = meshes_[rt.legendMesh].mesh;
    m.clear();
    TextStyle st;
    st.size = (k.style == KeyStyle::Number) ? 0.62f : 0.46f;
    if (text.size() == 1 && k.style != KeyStyle::Number) st.size = 0.6f;
    st.strokeWidth = k.style == KeyStyle::Number ? 0.15f : 0.14f;
    st.height = 0.022f;
    const float maxW = kKeySize * 0.74f;
    const float w = font_.measure(text) * st.size;
    if (w > maxW) st.size *= maxW / w;
    // texte centré, posé sur la face concave de la touche (coordonnées locales)
    const TextFrame frame{make_float3(0.0f, 0.0f, st.size * 0.5f), make_float3(1, 0, 0), make_float3(0, 0, -1),
                          make_float3(0, 1, 0)};
    RoundedBoxDesc local = capDesc_;
    local.center = make_float3(0, 0, 0);
    addText(m, font_, text, frame, st, TextAlign::Center,
            [local](float3 p) { return roundedBoxTopHeight(local, p.x, p.z); });
    meshes_[rt.legendMesh].dirty = true;
}

void CalculatorScene::setShift(bool shift)
{
    if (shift == shift_) return;
    shift_ = shift;
    for (size_t i = 0; i < keys_.size(); ++i)
        if (!keys_[i].altLabel.empty()) rebuildLegend(static_cast<int>(i));
}

void CalculatorScene::pressKey(int keyIndex)
{
    if (keyIndex < 0 || keyIndex >= static_cast<int>(keyRt_.size())) return;
    keyRt_[keyIndex].press = 1.0f;
    keyRt_[keyIndex].glow = 1.0f;
}

bool CalculatorScene::animate(float dt, int hoveredKey)
{
    bool changed = false;
    const float3 hoverColor = make_float3(0.25f, 0.9f, 0.35f);
    for (size_t i = 0; i < keyRt_.size(); ++i) {
        KeyRuntime& rt = keyRt_[i];
        const float prevPress = rt.press;
        rt.press = std::max(0.0f, rt.press - dt * 6.0f);
        const float targetGlow = (static_cast<int>(i) == hoveredKey) ? 0.35f : 0.0f;
        const bool shiftKey = keys_[i].id == KeyId::Second && shift_;
        const float goal = std::max(targetGlow, shiftKey ? 0.6f : 0.0f);
        const float prevGlow = rt.glow;
        rt.glow += (goal - rt.glow) * std::min(1.0f, dt * 12.0f);
        if (std::fabs(rt.glow - goal) < 1e-3f) rt.glow = goal;
        // enfoncement : courbe "ressort" (descente rapide, remontée amortie)
        const float p = rt.press;
        const float depth = kPressDepth * std::sin(p * kPi * 0.5f);
        const Affine xf = affineMul(calcXf_, affineTranslate(rt.center + make_float3(0, -depth, 0)));
        SceneInstance& cap = instances_[rt.capInstance];
        SceneInstance& leg = instances_[rt.legendInstance];
        cap.transform = xf;
        leg.transform = xf;
        const float3 gc = shiftKey ? make_float3(1.0f, 0.75f, 0.1f) : hoverColor;
        cap.glow = rt.glow;
        cap.glowColor = gc;
        leg.glow = rt.glow;
        leg.glowColor = gc;
        if (prevPress != rt.press || prevGlow != rt.glow) changed = true;
    }
    return changed;
}

void CalculatorScene::setDisplay(const DisplayContent& content)
{
    if (content == display_ && displayMainMesh_ >= 0) return;
    display_ = content;
    rebuildDisplay();
}

void CalculatorScene::rebuildDisplay()
{
    if (displayMainMesh_ < 0) {
        auto make = [&](int& meshId, int& instId, int material) {
            instId = addPart(Mesh{}, material);
            meshId = instances_[instId].mesh;
        };
        make(displayMainMesh_, displayMainInst_, kMatVfdBright);
        make(displayExprMesh_, displayExprInst_, kMatVfdDim);
        make(displayStatusMesh_, displayStatusInst_, kMatVfdStatus);
    }
    const float margin = 0.45f;
    const float left = panelCenter_.x - panelHalf_.x + margin;
    const float right = panelCenter_.x + panelHalf_.x - margin;
    const float back = panelCenter_.z - panelHalf_.y;
    const float front = panelCenter_.z + panelHalf_.y;
    const float maxW = right - left;
    auto frameAt = [&](float x, float zBaseline) {
        return TextFrame{make_float3(x, panelTop_, zBaseline), make_float3(1, 0, 0), make_float3(0, 0, -1),
                         make_float3(0, 1, 0)};
    };
    auto fit = [&](const std::string& s, TextStyle& st, float width) {
        const float w = font_.measure(s) * st.size;
        if (w > width) st.size *= width / w;
    };

    // ---- ligne principale (grands chiffres VFD, alignés à droite)
    {
        Mesh& m = meshes_[displayMainMesh_].mesh;
        m.clear();
        TextStyle st;
        st.size = 1.2f;
        st.strokeWidth = 0.13f;
        st.height = 0.03f;
        fit(display_.main, st, maxW);
        addText(m, font_, display_.main, frameAt(right, front - 0.42f), st, TextAlign::Right);
        meshes_[displayMainMesh_].dirty = true;
        instances_[displayMainInst_].material =
            display_.error ? kMatVfdError : (display_.mainIsPreview ? kMatVfdDim : kMatVfdBright);
    }
    // ---- ligne d'expression (affiche la fin si trop longue)
    {
        Mesh& m = meshes_[displayExprMesh_].mesh;
        m.clear();
        TextStyle st;
        st.size = 0.5f;
        st.strokeWidth = 0.13f;
        st.height = 0.025f;
        std::string e = display_.expression;
        while (!e.empty() && font_.measure(e) * st.size > maxW) e.erase(e.begin());
        addText(m, font_, e, frameAt(left, back + 1.25f), st, TextAlign::Left);
        meshes_[displayExprMesh_].dirty = true;
    }
    // ---- indicateurs d'état
    {
        Mesh& m = meshes_[displayStatusMesh_].mesh;
        m.clear();
        TextStyle st;
        st.size = 0.28f;
        st.strokeWidth = 0.15f;
        st.height = 0.02f;
        addText(m, font_, display_.statusLeft, frameAt(left, back + 0.5f), st, TextAlign::Left);
        addText(m, font_, display_.statusRight, frameAt(right, back + 0.5f), st, TextAlign::Right);
        meshes_[displayStatusMesh_].dirty = true;
    }
}

}  // namespace crtx
