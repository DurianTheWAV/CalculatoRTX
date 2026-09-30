// CalculatoRTX - capture d'écran BMP.
#include "Screenshot.h"

#include <cstdio>

namespace crtx {

bool writeBmp(const std::string& path, const std::vector<uint32_t>& pixels, uint32_t width, uint32_t height, bool bgra)
{
    if (pixels.size() < static_cast<size_t>(width) * height || width == 0 || height == 0) return false;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t rowBytes = (width * 3 + 3) & ~3u;
    const uint32_t dataBytes = rowBytes * height;
    unsigned char header[54] = {'B', 'M'};
    auto put32 = [&](int at, uint32_t v) {
        for (int i = 0; i < 4; ++i) header[at + i] = static_cast<unsigned char>(v >> (8 * i));
    };
    put32(2, 54 + dataBytes);
    put32(10, 54);
    put32(14, 40);
    put32(18, width);
    put32(22, height);
    header[26] = 1;   // plans
    header[28] = 24;  // bits par pixel
    put32(34, dataBytes);
    put32(38, 2835);  // 72 ppp
    put32(42, 2835);
    bool ok = std::fwrite(header, 1, 54, f) == 54;
    std::vector<unsigned char> row(rowBytes, 0);
    for (uint32_t y = 0; y < height && ok; ++y) {
        const uint32_t* src = pixels.data() + static_cast<size_t>(height - 1 - y) * width;  // BMP : de bas en haut
        for (uint32_t x = 0; x < width; ++x) {
            const uint32_t p = src[x];
            const unsigned char c0 = p & 0xFF, c1 = (p >> 8) & 0xFF, c2 = (p >> 16) & 0xFF;
            // BMP attend B, G, R
            row[x * 3 + 0] = bgra ? c0 : c2;
            row[x * 3 + 1] = c1;
            row[x * 3 + 2] = bgra ? c2 : c0;
        }
        ok = std::fwrite(row.data(), 1, rowBytes, f) == rowBytes;
    }
    std::fclose(f);
    return ok;
}

}  // namespace crtx
