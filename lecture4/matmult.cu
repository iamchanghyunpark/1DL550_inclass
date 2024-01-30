#include <stdio.h>
#include "common_cuda.h"

__global__ void MatMult(int *A, int *B, int *C, int N)
{
    //Multiply A and B and store into C.
    //Do matrix matrix multiplication (not per-element multiplication)
}

int main(int argc, const char **argv)
{
    int n = 16;

    if (argc == 2)
        n = atoi(argv[1]);
    printf("Size of one dimension of matrix is %d\n", n);
    printf("Size of matrix: %d\n", n* n* sizeof(int));

    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    size_t bytes = n*n*sizeof(int);

    //1. Allocate the host buffers h_A, h_B, h_C;

    for (int i = 0; i < n*n; i++) {
        h_A[i] = i;
    }
    memset(h_B, 0, bytes);
    memset(h_C, 0, bytes);

    for (int i = 0; i < n; i++) {
        h_B[i * n + i] = 1;
    } // Make Identity matrix
    //Easier to check later on!

    //2. allocate the device buffers d_A, d_B, d_C

    //3. Copy the values in the host buffers into the device buffers

    //4. Calculate the number of threads, and number of blocks.
    // Hint: you could go 1D or even 2D! If you go 1D, 1024 threads per block. If you go 2D: 16x16 threads per block! (Or maybe even 32x32)

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //5. Invoke the CUDA kernel!
    cudaEventRecord(stop);

    //6. Copy the results back from the device into the host

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel took %fms\n", milliseconds);

    //7. If you want to validate your calculations, compare element by element whether A and C are equivalent!
}
