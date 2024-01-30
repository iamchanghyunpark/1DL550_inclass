#include "ppm_cuda.h"

//Host buffers
unsigned char *unified_buffer;
unsigned char *unified_grayscale_buffer;

//Corresponding Device buffers
unsigned char *d_unified_buffer;
unsigned char *d_unified_grayscale_buffer;

__global__ void grayscale_cuda (unsigned char *img_data, unsigned char *grayscale_data, size_t lim) {
    // Fill in this kernel!

    //1. calculate the index that this thread is responsible for. (Don't forget blockDim and blockIdx!)

    //2. Calculate the grayscale value of each pixel. Remember, a thread will be responsible for one pixel!
    //3. Did you remember to take care of the situation where the thread idx may be out of the image bound?
    //Remember to have an if statement to make sure we are within image size bounds.
    //4. Fill in the computed gray-scale value into the grayscale_data array. (at the correct index ;)
}

void usage(char **argv) {
        printf("Usage: %s <ppm-file1> [<ppm-file2> ...]\n", argv[0]);
}

void allocateAllBuffers(PPMImage **ppm_list, int len)
{
    size_t total_bytes = 0;
    size_t grayscale_total_bytes = 0;
    for (int i = 0; i < len; i++) {
        grayscale_total_bytes += ppm_list[i]->img_size;
        total_bytes += ppm_list[i]->img_size * 3;
    }

    if (cudaMallocHost(&unified_buffer, total_bytes) != cudaSuccess) {
        printf("Error allocating host paged memory for the unified buffer\n");
        return;
    }

    if (cudaMalloc(&d_unified_buffer, total_bytes) != cudaSuccess) {
        printf("Error allocating device paged memory for the unified buffer\n");
        return;
    }

    if (cudaMallocHost(&unified_grayscale_buffer, grayscale_total_bytes) != cudaSuccess) {
        printf("Error allocating host paged memory for the unified buffer\n");
        return;
    }

    if (cudaMalloc(&d_unified_grayscale_buffer, grayscale_total_bytes) != cudaSuccess) {
        printf("Error allocating device paged memory for the unified buffer\n");
        return;
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv);
        return 1;
    }

    PPMImage **ppm_list = (PPMImage **)malloc(sizeof(PPMImage*) * (argc-1));
    for (int i = 1; i < argc; i++) {
        ppm_list[i-1] = parseWithoutAllocatingBuffersPPM(argv[i]);
    }
    allocateAllBuffers(ppm_list, argc-1);

    size_t progress = 0;
    for (int i = 0; i < argc-1; i++) {
        ppm_list[i]->buffer_start =  progress;
        ppm_list[i]->grayscale_buffer_start = (progress/3);

        progress += readIntoBuffersPPM(ppm_list[i], argv[i+1], unified_buffer);
    }

    printf("Processing %d images\n", argc-1);

    struct timeval start, end;
    double msecs;

    gettimeofday(&start, NULL);
    /** Fill the following four things!!! */

    // 1. the unified_buffer has now been filled. Copy this into the device buffer (d_unified_buffer).
    // Hint: You should copy `progress` number of bytes.


    for (int i = 0; i < argc - 1; i++) {
        PPMImage *ppm = ppm_list[i];

        //2. Now for each image (described by the ppm) try to divide up the image data into different CUDA blocks.
        //For now, I suggest sticking with 1024 threads per block. (You could try smaller blocks too and compare performance)
        //(hint: ppm->img_size provides total number of pixels in the image)
        //Remember to handle the remainders. Number of pixel will not be completely divisable by 1024. You need to handle the remainders too.


        //3. Calling the cuda kernel
        //Fill in the two FILL_MEs. I have filled in the other arguments.
        //First argument: passes the pointer that points to the image's first pixel.
        //Second argument: passes the pointer to the grayscale buffer to be filled in by the CUDA kernel
        //Third argument: total number of pixels in the image.
        grayscale_cuda<<<FILL_ME, FILL_ME>>>(d_unified_buffer + ppm->buffer_start, d_unified_grayscale_buffer + ppm->grayscale_buffer_start, ppm->img_size);
    }
    //4. Copy the grayscale buffer values from the device buffer into the host buffer.
    //Use the `progress` variable to calculate the number of bytes to copy.
    //Do you want to copy the same number of bytes as in field number 1 above?


    gettimeofday(&end, NULL);
    msecs = get_time_diff_ms(start, end);
    printf("Processing %d images in CUDA took %lf ms\n", argc-1, msecs);

    for (int i = 0; i < argc -1; i++) {
       writePGMFromBuffer(ppm_list[i], argv[i+1], unified_grayscale_buffer+ppm_list[i]->grayscale_buffer_start);
        freePPM(ppm_list[i]);
    }

    return 0;
}
