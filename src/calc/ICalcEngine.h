// CalculatoRTX - interface du moteur de calcul GPU.
//
// Deux implémentations, toutes deux exécutées sur la carte graphique :
//   * CalcEngine   (CUDA, backend NVIDIA)      : src/calc/CalcEngine.cu
//   * VkCalcEngine (Vulkan compute, AMD/Intel/NVIDIA) : src/vk/VkCalcEngine.cpp
// L'hôte ne fait qu'encoder l'expression saisie en octets (Request) et relire le texte
// formaté par le GPU (Result).
#pragma once

#include "CalcTypes.h"

#include <string>

namespace crtx {
namespace calc {

class ICalcEngine {
public:
    virtual ~ICalcEngine() = default;

    // Envoie le programme au GPU et attend le résultat (quelques dizaines de µs).
    virtual Result evaluate(const std::string& program, AngleMode mode, double ansHi, double ansLo, double memHi,
                            double memLo, float* gpuMicroseconds = nullptr) = 0;

    // Nom court affiché dans l'état de l'afficheur ("CUDA", "VK").
    virtual const char* apiName() const = 0;
};

// Construit la requête transmise au GPU (copie d'octets, aucun calcul).
Request makeRequest(const std::string& program, AngleMode mode, double ansHi, double ansLo, double memHi,
                    double memLo);

// Auto-test exécuté au démarrage (quelques expressions de référence évaluées par le GPU).
bool runSelfTest(ICalcEngine& engine);

}  // namespace calc
}  // namespace crtx
