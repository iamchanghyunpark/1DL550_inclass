#ifndef PPM_H
#define PPM_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "common.h"

#ifdef __arm__
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#include <emmintrin.h>
#else
#error "Unsupported Architecture"
#endif

typedef struct {
    char type[3];
    size_t width;
    size_t height;
    size_t img_size; // width * height
    int maxval;
    float* data; // Pointer to the image data
    float* grayscale_data; // Pointer to the grayscale image data
#ifdef __arm__
    //Arm provides fp16. Try working on fp16!
    float16_t* data_fp16;
    float16_t* grayscale_data_fp16;
#endif
} PPMImage;

PPMImage* parsePPM(const char* filename);
void writePGM(const PPMImage* ppm, const char* original_filename);
void freePPM(PPMImage* ppm);

void getRGBArrays(float **R, float **G, float **B, PPMImage *ppm);

#ifdef __arm__
void getRGBArrays_fp16(float16_t **R, float16_t **G, float16_t **B, PPMImage *ppm);
#endif

#endif /* PPM_H */
