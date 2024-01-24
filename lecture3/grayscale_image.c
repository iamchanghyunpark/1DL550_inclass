#include "ppm.h"

void grayscale_image_serial(PPMImage *ppm)
{
    float *R, *G, *B;
    getRGBArrays(&R, &G, &B, ppm);

    long i;
    for (i = 0; i < ppm->img_size; i++) {
        float sum = 0.0;
        sum += R[i] * 0.21;
        sum += G[i] * 0.72;
        sum += B[i] * 0.07;
        ppm->grayscale_data[i] = sum;
    }
}

#ifdef __x86_64__
void grayscale_image_simd_avx(PPMImage *ppm)
{
    float *R, *G, *B;
    getRGBArrays(&R, &G, &B, ppm);

    //Fill this code
}

void grayscale_image_simd_sse(PPMImage *ppm)
{
    float *R, *G, *B;
    getRGBArrays(&R, &G, &B, ppm);

    //Fill this code
}

#elif defined(__arm__)
void grayscale_image_simd_neon(PPMImage *ppm)
{
    float *R, *G, *B;
    getRGBArrays(&R, &G, &B, ppm);

    //Fill this code
}

void grayscale_image_simd_neon_fp16(PPMImage *ppm)
{
    float16_t *R, *G, *B;
    getRGBArrays_fp16(&R, &G, &B, ppm);

    //Fill this code
}
#endif

void usage(char **argv) {
        printf("Usage: %s type_of_processing <ppm-file1> [<ppm-file2> ...]\n", argv[0]);
#ifdef __x86_64__
        printf("Type of processing: 'serial', 'avx', 'sse'\n");
#elif defined(__arm__)
        printf("Type of processing: 'serial', 'neon', 'neon_fp16'\n");
#endif
}

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv);
        return 1;
    }

    void (*process_image)(PPMImage *) = NULL;
    char processingType[20];

    if (strcmp(argv[1], "serial") == 0) {
        process_image = grayscale_image_serial;
        sprintf(processingType, "serial");
    }
#ifdef __x86_64__
    else if (strcmp(argv[1], "avx") == 0) {
        process_image = grayscale_image_simd_avx;
        sprintf(processingType, "avx");
    }
    else if (strcmp(argv[1], "sse") == 0) {
        process_image = grayscale_image_simd_sse;
        sprintf(processingType, "sse");
    }
#elif defined(__arm__)
    else if (strcmp(argv[1], "neon") == 0) {
        process_image = grayscale_image_simd_neon;
        sprintf(processingType, "neon");
    }
    else if (strcmp(argv[1], "neon_fp16") == 0) {
        process_image = grayscale_image_simd_neon_fp16;
        sprintf(processingType, "neon_fp16");
    }
#endif
    else {
        printf("Unknown processing type. You provided %s.\n", argv[1]);
        usage(argv);
        exit(1);
    }

    PPMImage **ppm_list = malloc(sizeof(PPMImage*) * (argc-2));
    for (int i = 2; i < argc; i++) {
        ppm_list[i-2] = parsePPM(argv[i]);
//        printf("[%d] File: %s\nType: %s\nResolution: %ldx%ld\nMax value per channel: %d\n",
//               i-2, argv[i], ppm_list[i-2]->type, ppm_list[i-2]->width, ppm_list[i-2]->height, ppm_list[i-2]->maxval);
    }

    printf("Processing %d images\n", argc-2);

    double serial_total = 0.0;
    for (int i = 0; i < argc-2; i++) {
        if (ppm_list[i]) {
            struct timeval start,end;
            double msecs;

            gettimeofday(&start, NULL);
            process_image(ppm_list[i]);
            gettimeofday(&end, NULL);
            msecs = get_time_diff_ms(start, end);
            printf("[%d]: %lf ms\n", i, msecs);
            serial_total += msecs;
        }
    }

    printf("%s took a total of %lf ms, average %lf ms\n", processingType, serial_total, serial_total/(argc-1));

    for (int i = 0; i < argc -2; i++) {
        writePGM(ppm_list[i], argv[i+2]);
        freePPM(ppm_list[i]);
    }

    return 0;
}
