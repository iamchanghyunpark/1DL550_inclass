#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "common_cuda.h"

#ifndef SIZE
#define SIZE 100000000L
#endif

#define THREADSPERBLOCK 1024

float *X, *Y;
float *d_X, *d_Y;
long *d_perBlockSum;
long *perBlockSum;
size_t blocks;

void init() {
  // Fill me! Initialize buffers (both host and device)

  // 1. Allocate host buffers for random X values and random Y values
  // size of each buffers will be SIZE number of elements. (hint: don't forget sizeof(TYPE))
  // 2. Also allocate the device buffers for X and Y (d_X, d_Y)

  //Fill X and Y with random numbers [0, 1]
  for (int i = 0; i < SIZE; i++) {
      X[i] = (double)rand()/(double)RAND_MAX;
      Y[i] = (double)rand()/(double)RAND_MAX;
  }

  // 3. We also need an array that will hold per-(CUDA)-block values (long *perBlockSum, *d_perBlockSum)
  // Calculate how many blocks you will be running, assuming you will be evaluating SIZE number of elements and 1024 threads per block (or less if you want to explore.)
  // then allocate both host and device buffers.
}

__global__ void pi_cuda(float *d_X, float *d_Y, long *perBlockSum, int n) {
    extern __shared__ int sharedData[]; //We will learn about this later on.
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (/* bounds check whether we are within bounds */) {
        // Calculate whether this particular index, i is within a distance of 1.0 from the origin (0,0).
        if (/*If the distance is within 1.0 */)
            sharedData[tid] = 1;
        else // If distance exceeds 1.0
            sharedData[tid] = 0;
    } else { // If we our out of bounds
        sharedData[tid] = 0;
    }
    __syncthreads();

    //Intra block reduction
    //What do you think this is doing?
    for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid+s];
        }
        __syncthreads();
    }

    //For the first threads of each block, store the sum of this entire block into the perBlockSum!
    if (tid == 0) {
        perBlockSum[blockIdx.x] = sharedData[0];
    }
}

int main(int argc, char *argv[]) {
  init();

  double pi;
  struct timeval start, end;
  long sum = 0;

  gettimeofday(&start, NULL);
  // 1. Copy the random values from the X and Y host buffers into corresponding device buffers
  // 2. Calculate the number of blocks (you can reuse the values from your calculation in init(), or redo it here)
  // 3. Based on the number of blocks you calculated, fill in the cuda kernel invocation (specifically the FILL_MEs)
  // First two arguments: X and Y (device buffers)
  // third argument: device buffer to hold per-block counts
  // fourth argument: total number of coordinates to test
  pi_cuda<<<FILL_ME, FILL_ME, THREADSPERBLOCK*sizeof(int)>>>(d_X, d_Y, d_perBlockSum, SIZE);

  // 4. Copy the per-block count values back from the device to host.

  // Loop through all the perBlock values and sum it up.
  // If you have time and want to be adventurous, you could also do this in CUDA too!
  for (int i = 0; i < blocks ; i++) {
      sum += perBlockSum[i];
  }
  pi = (double)sum / (double)SIZE * 4;
  gettimeofday(&end, NULL);
  printf("Execution time of cuda: %lfms\n", get_time_diff_ms(start, end));
  printf("Pi: %lf\n", pi);

  return 0;
}
