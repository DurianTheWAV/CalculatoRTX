// CalculatoRTX - capture d'écran au format BMP 24 bits (lisible partout, sans dépendance).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace crtx {

// pixels : 1 uint32 par pixel, octets B,G,R,A (bgra = true) ou R,G,B,A en mémoire.
bool writeBmp(const std::string& path, const std::vector<uint32_t>& pixels, uint32_t width, uint32_t height, bool bgra);

}  // namespace crtx
