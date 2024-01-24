#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "common.h"

#ifdef __arm__
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <emmintrin.h>
#include <immintrin.h>
#else
#error "Unknown platform!"
#endif

#ifndef SIZE
#define SIZE 100000000L
#endif

float *X, *Y;

void init() {
  //Using posix_memalign instead of _mm_malloc() just for portability.
  //They do the same thing by aligning at 32B.
  posix_memalign((void **)&X, MALLOC_ALIGN, SIZE*sizeof(float));
  posix_memalign((void **)&Y, MALLOC_ALIGN, SIZE*sizeof(float));

  //Fill X and Y with random numbers [0, 1]
  for (int i = 0; i < SIZE; i++) {
      X[i] = (double)rand()/(double)RAND_MAX;
      Y[i] = (double)rand()/(double)RAND_MAX;
  }
}

double pi_novec()
{
  long count = 0;

  for (int i = 0; i < SIZE; i++) {
      double val = X[i]*X[i] + Y[i]*Y[i];
      val = sqrt(val);
      if (val <= 1.0)
          count++;
  }
  return (double)count/SIZE*4.0;
}

#ifdef __arm__
double pi_neon()
{
  long long count = 0;

  //Fill in this code

  return (double)count/SIZE*4.0;
}
#elif defined(__x86_64__)
double pi_sse() {
  long count = 0;

  //Fill in this code

  return (double)count/SIZE*4.0;
}

double pi_avx() {
  long count = 0;

  //Fill in this code

  return (double)count/SIZE*4.0;
}
#endif

void usage(char **argv) {
        printf("Usage: %s type_of_processing \n", argv[0]);
#ifdef __x86_64__
        printf("Type of processing: 'serial', 'avx', 'sse'\n");
#elif defined(__arm__)
        printf("Type of processing: 'serial', 'neon'\n");
#endif
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage(argv);
        return 1;
    }
  init();

  double (*pi_type)(void) = NULL;
  char processingType[20];

    if (strcmp(argv[1], "serial") == 0) {
        pi_type = pi_novec;
        sprintf(processingType, "serial");
    }
#ifdef __x86_64__
    else if (strcmp(argv[1], "avx") == 0) {
        pi_type = pi_avx;
        sprintf(processingType, "avx");
    }
    else if (strcmp(argv[1], "sse") == 0) {
        pi_type = pi_sse;
        sprintf(processingType, "sse");
    }
#elif defined(__arm__)
    else if (strcmp(argv[1], "neon") == 0) {
        pi_type = pi_neon;
        sprintf(processingType, "neon");
    }
#endif
    else {
        printf("Unknown processing type. You provided %s.\n", argv[1]);
        usage(argv);
        exit(1);
    }


  double pi;
  struct timeval start, end;

  gettimeofday(&start, NULL);
  pi = pi_type();
  gettimeofday(&end, NULL);
  printf("Execution time of %s: %lfms\n", processingType, get_time_diff_ms(start, end));
  printf("Pi: %lf\n", pi);

  return 0;
}
