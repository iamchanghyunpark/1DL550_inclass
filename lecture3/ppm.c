#include "ppm.h"

PPMImage* parsePPM(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    PPMImage* ppm = malloc(sizeof(PPMImage));
    if (!ppm) {
        perror("Failed to allocate memory for PPMImage");
        fclose(file);
        return NULL;
    }
    memset(ppm, 0, sizeof(PPMImage));

    // Read the type of PPM (P3 or P6)
    if (fscanf(file, "%2s", ppm->type) != 1) {
        printf("Failed to read type\n");
        fclose(file);
        free(ppm);
        return NULL;
    }

    // Skip whitespaces and potential comments
    int ch;
    do {
        ch = fgetc(file);
        if (ch == '#') { // Skip comments
            while ((ch = fgetc(file)) != '\n' && ch != EOF);
        }
    } while (isspace(ch) && ch != EOF);

    // We've read one char too many, step back
    if (ch != EOF) {
        ungetc(ch, file);
    }

    // Read resolution
    if (fscanf(file, "%ld %ld", &ppm->width, &ppm->height) != 2) {
        printf("Failed to read resolution\n");
        fclose(file);
        free(ppm);
        return NULL;
    }

    // Skip whitespaces and potential comments again
    do {
        ch = fgetc(file);
        if (ch == '#') { // Skip comments
            while ((ch = fgetc(file)) != '\n' && ch != EOF);
        }
    } while (isspace(ch) && ch != EOF);

    // Step back one char as before
    if (ch != EOF) {
        ungetc(ch, file);
    }

    // Read max value per channel
    if (fscanf(file, "%d", &ppm->maxval) != 1) {
        printf("Failed to read max value per channel\n");
        fclose(file);
        free(ppm);
        return NULL;
    }

    // The next byte after a whitespace is the start of image data
    do {
        ch = fgetc(file);
    } while (isspace(ch) && ch != EOF);

    // Step back one char as before
    if (ch != EOF) {
        ungetc(ch, file);
    }

    // Calculate the size of the image data
    size_t data_size;
    if (strcmp(ppm->type, "P6") == 0 && ppm->maxval == 255) {
        ppm->img_size = ppm->width * ppm->height;
        data_size = ppm->img_size * 3;
    } else {
        // Handling P3 format is more complex due to ASCII values.
        // It requires parsing the entire file to determine the size.
        // For simplicity, this example does not handle P3 completely.
        // Also added a requirement for maxval to be 255... 
        printf("%s format not fully supported with maxval=%d\n", ppm->type, ppm->maxval);
        fclose(file);
        free(ppm);
        return NULL;
    }

    unsigned char *raw_bytes = malloc(data_size);
    if (!raw_bytes) {
        printf("Error allocating raw_bytes\n");
        fclose(file);
        free(ppm);
        return NULL;
    }

    // Allocate memory for image data
    if (posix_memalign((void **)&ppm->data, MALLOC_ALIGN, data_size*sizeof(float))) {
        perror("Failed to allocate memory for image data");
        fclose(file);
        free(ppm);
        return NULL;
    }
#ifdef __arm__
    // Allocate memory for image data fp16
    if (posix_memalign((void **)&ppm->data_fp16, MALLOC_ALIGN, data_size*sizeof(float16_t))) {
        perror("Failed to allocate memory for image data");
        fclose(file);
        free(ppm);
        return NULL;
    }
#endif

    // Read the image data
    size_t total_read = 0;
    while (total_read < data_size) {
        size_t bytes_read = fread(raw_bytes + total_read, 1, data_size - total_read, file);
        if (bytes_read == 0) {
            if (feof(file)) {
                printf("End of file reached unexpectedly\nRead %ld bytes out of %ld expected.", total_read, data_size);
                break;
            }
            if (ferror(file)) {
                perror("Error reading file");
                fclose(file);
                free(ppm->data);
                free(raw_bytes);
                free(ppm);
                return NULL;
            }
        }
        total_read += bytes_read;
    }
    //Moving Raw into actual ppm->data
    float *R, *G, *B;
    getRGBArrays(&R, &G, &B, ppm);

#ifdef __arm__
    float16_t *R_fp16, *G_fp16, *B_fp16;
    getRGBArrays_fp16(&R_fp16, &G_fp16, &B_fp16, ppm);
#endif

    for (size_t i = 0; i < ppm->img_size; i++) {
        R[i] = (float)raw_bytes[i*3];
        G[i] = (float)raw_bytes[i*3+1];
        B[i] = (float)raw_bytes[i*3+2];
#ifdef __arm__
        R_fp16[i] = (float16_t)raw_bytes[i*3];
        G_fp16[i] = (float16_t)raw_bytes[i*3+1];
        B_fp16[i] = (float16_t)raw_bytes[i*3+2];
#endif
    }
    free(raw_bytes);

    //Allocating grayscale image data
    data_size = ppm->img_size * sizeof(float);
    if (posix_memalign((void **)&ppm->grayscale_data, MALLOC_ALIGN, data_size)) {
        perror("Failed to allocate memory for grayscale image data");
        fclose(file);
        free(ppm->data);
        free(ppm);
        return NULL;
    }
    memset(ppm->grayscale_data, 0, data_size);

#ifdef __arm__
    data_size = ppm->img_size * sizeof(float16_t);
    if (posix_memalign((void **)&ppm->grayscale_data_fp16, MALLOC_ALIGN, data_size)) {
        perror("Failed to allocate memory for grayscale image data_fp16");
        fclose(file);
        free(ppm->data);
        free(ppm);
        return NULL;
    }
    memset(ppm->grayscale_data_fp16, 0, data_size);
#endif


    fclose(file);
    return ppm;
}

void writePGM(const PPMImage* ppm, const char* original_filename) {
    if (!ppm || !original_filename) {
        printf("Invalid arguments to writePGM\n");
        return;
    }

    // Change file extension from .ppm to .pgm
    char* new_filename = malloc(strlen(original_filename) + 1 + 4); // +1 for null terminator
    if (!new_filename) {
        perror("Failed to allocate memory for new filename");
        return;
    }
    strcpy(new_filename, original_filename);
    char* ext = strrchr(new_filename, '.');
    if (ext != NULL) {
        strcpy(ext, ".pgm");
    } else {
        strcat(new_filename, ".pgm");
    }

#ifdef __arm__
    // Change file extension from .ppm to .pgm
    char* new_filename_fp16 = malloc(strlen(new_filename) + 1 + 9); // +1 for null terminator
    if (!new_filename_fp16) {
        perror("Failed to allocate memory for new filename");
        return;
    }
    strcpy(new_filename_fp16, new_filename);
    ext = strrchr(new_filename_fp16, '.');
    if (ext != NULL) {
        strcpy(ext, "_fp16.pgm");
    } else {
        strcat(new_filename_fp16, "_fp16.pgm");
    }

    FILE* file_fp16 = fopen(new_filename_fp16, "wb");
    if (!file_fp16) {
        perror("Error opening file_fp16");
        free(new_filename);
        return;
    }
    fprintf(file_fp16, "P5\n%ld %ld\n%d\n", ppm->width, ppm->height, ppm->maxval);
#endif

    FILE* file = fopen(new_filename, "wb");
    if (!file) {
        perror("Error opening file");
        free(new_filename);
        return;
    }

    // Write the PGM header
    fprintf(file, "P5\n%ld %ld\n%d\n", ppm->width, ppm->height, ppm->maxval);

    size_t img_size = ppm->width * ppm->height;
    unsigned char *raw_data = malloc(img_size);
    if (!raw_data) {
        printf("Error writing back! Cannot allocate raw_data array\n");
        exit(1);
    }

#ifdef __arm__
    unsigned char *raw_data_fp16 = malloc(img_size);
    if (!raw_data_fp16) {
        printf("Error writing back! Cannot allocate raw_data_fp16 array\n");
        exit(1);
    }
#endif

    for (size_t i = 0; i < img_size; i++) {
        raw_data[i] = (unsigned char)ppm->grayscale_data[i];
#ifdef __arm__
        raw_data_fp16[i] = (unsigned char)ppm->grayscale_data_fp16[i];
#endif
    }

    // Write the grayscale image data with a loop
    size_t total_written = 0;
    while (total_written < img_size) {
        size_t written = fwrite(raw_data + total_written, 1, img_size - total_written, file);
        if (written == 0) {
            if (ferror(file)) {
                perror("Error writing grayscale data to file");
                break;
            }
        }
        total_written += written;
    }

#ifdef __arm__
    // Write the grayscale image data with a loop
    total_written = 0;
    while (total_written < img_size) {
        size_t written = fwrite(raw_data_fp16 + total_written, 1, img_size - total_written, file_fp16);
        if (written == 0) {
            if (ferror(file_fp16)) {
                perror("Error writing grayscale data_fp16 to file_fp16");
                break;
            }
        }
        total_written += written;
    }
    fclose(file_fp16);
    free(new_filename_fp16);
#endif

    fclose(file);
    free(new_filename);
    free(raw_data);
}

void freePPM(PPMImage* ppm) {
    if (ppm) {
        if (ppm->data) {
            free(ppm->data);
        }
        if (ppm->grayscale_data) {
            free(ppm->grayscale_data);
        }
        free(ppm);
    }
}

void getRGBArrays(float **R, float **G, float **B, PPMImage *ppm)
{
    *R = ppm->data;
    *G = &ppm->data[ppm->img_size];
    *B = &ppm->data[ppm->img_size*2];
}

#ifdef __arm__
void getRGBArrays_fp16(float16_t **R, float16_t **G, float16_t **B, PPMImage *ppm)
{
    *R = ppm->data_fp16;
    *G = &ppm->data_fp16[ppm->img_size];
    *B = &ppm->data_fp16[ppm->img_size*2];
}
#endif

