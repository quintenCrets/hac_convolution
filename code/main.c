#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>

void applyConvolution(unsigned char* image, unsigned char* output, int width, int height, int channels, float kernel[3][3]);
void applyMaxPooling(unsigned char* image, unsigned char* output, int width, int height, int channels);
void applyAveragePooling(unsigned char* image, unsigned char* output, int width, int height, int channels);

int main(int argc, char* argv[]) {
    char* filename;
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        filename = "Bird.jpeg";

    }
    else {
        filename = argv[1];
    }

    int width, height, channels;
    unsigned char* img = stbi_load(filename, &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image: %s\n", stbi_failure_reason());
        printf("Error in loading the image\n");
        return -1;
    }

    // Convolution
    float kernel[3][3] = {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}};
    unsigned char* convOutput = (unsigned char*)malloc(width * height * channels);
    applyConvolution(img, convOutput, width, height, channels, kernel);
    stbi_write_png("C:/Users/thoma/Documents/QT_projects/Convolutie/conv_output.jpeg", width, height, channels, convOutput, width * channels);
    printf("Finished Convolution \n\r");

    // Max Pooling
    unsigned char* maxPoolOutput = (unsigned char*)malloc((width / 2) * (height / 2) * channels);
    applyMaxPooling(img, maxPoolOutput, width, height, channels);
    stbi_write_png("C:/Users/thoma/Documents/QT_projects/Convolutie/max_pool_output.jpeg", width / 2, height / 2, channels, maxPoolOutput, (width / 2) * channels);
    printf("Finished Max pooling \n\r");


    // Average Pooling
    unsigned char* avgPoolOutput = (unsigned char*)malloc((width / 2) * (height / 2) * channels);
    applyAveragePooling(img, avgPoolOutput, width, height, channels);
    stbi_write_png("C:/Users/thoma/Documents/QT_projects/Convolutie/avg_pool_output.jpeg", width / 2, height / 2, channels, avgPoolOutput, (width / 2) * channels);
    printf("Finished Average pooling \n\r");

    // Cleanup
    stbi_image_free(img);
    free(convOutput);
    free(maxPoolOutput);
    free(avgPoolOutput);

    return 0;
}

void applyConvolution(unsigned char* image, unsigned char* output, int width, int height, int channels, float kernel[3][3]) {
    int edge = 1; // Since kernel size is 3x3

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum[3] = {0.0, 0.0, 0.0}; // Sum for each channel

            for (int ky = -edge; ky <= edge; ky++) {
                for (int kx = -edge; kx <= edge; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        for (int ch = 0; ch < channels; ch++) {
                            if (ch < 3) { // Apply convolution only to RGB channels
                                sum[ch] += kernel[ky + edge][kx + edge] * image[(iy * width + ix) * channels + ch];
                            }
                        }
                    }
                }
            }
            for (int ch = 0; ch < channels; ch++) {
                if (ch < 3) {
                    int val = (int)sum[ch];
                    output[(y * width + x) * channels + ch] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
                } else {
                    // Preserve the alpha channel if present
                    output[(y * width + x) * channels + ch] = image[(y * width + x) * channels + ch];
                }
            }
        }
    }
}

void applyMaxPooling(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    int outputWidth = width / 2;
    int outputHeight = height / 2;

    for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
            for (int ch = 0; ch < channels; ch++) {
                unsigned char maxVal = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int iy = y * 2 + dy;
                        int ix = x * 2 + dx;
                        unsigned char val = image[(iy * width + ix) * channels + ch];
                        if (val > maxVal) maxVal = val;
                    }
                }
                output[(y * outputWidth + x) * channels + ch] = maxVal;
            }
        }
    }
}


void applyAveragePooling(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    int outputWidth = width / 2;
    int outputHeight = height / 2;

    for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
            for (int ch = 0; ch < channels; ch++) {
                unsigned int sum = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int iy = y * 2 + dy;
                        int ix = x * 2 + dx;
                        sum += image[(iy * width + ix) * channels + ch];
                    }
                }
                output[(y * outputWidth + x) * channels + ch] = sum / 4;
            }
        }
    }
}
