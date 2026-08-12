#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using matrixd = std::vector<std::vector<double>>;
using vectord = std::vector<double>;

struct Image {
    int width = 0;
    int height = 0;

    matrixd r;
    matrixd g;
    matrixd b;

    static Image loadImage(const std::string& filename) {
        int w, h, channels;

        unsigned char* data = stbi_load(filename.c_str(), &w, &h, &channels, 3);

        if (!data) {
            std::cerr << "Failed to load image: " << filename << "\n";
            return {};
        }

        Image img;
        img.width = w;
        img.height = h;
        img.r.assign(h, vectord(w, 0.0));
        img.g.assign(h, vectord(w, 0.0));
        img.b.assign(h, vectord(w, 0.0));

      for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int idx = (i * w + j) * 3;
                img.r[i][j] = static_cast<double>(data[idx + 0]);
                img.g[i][j] = static_cast<double>(data[idx + 1]);
                img.b[i][j] = static_cast<double>(data[idx + 2]);
            }
        }

        stbi_image_free(data);
        return img;

    }

    static bool saveJPG(const std::string& filename,
                        const matrixd& r, const matrixd& g, const matrixd& b) {

        int h = r.size();
        int w = r[0].size();
        std::vector<unsigned char> bytes(w * h * 3);

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int idx = (i * w + j) * 3;
                bytes[idx + 0] = static_cast<unsigned char>(std::clamp(r[i][j], 0.0, 255.0));
                bytes[idx + 1] = static_cast<unsigned char>(std::clamp(g[i][j], 0.0, 255.0));
                bytes[idx + 2] = static_cast<unsigned char>(std::clamp(b[i][j], 0.0, 255.0));

            }
        }
            return stbi_write_jpg(filename.c_str(), w, h, 3, bytes.data(), 100) != 0;
        };

};