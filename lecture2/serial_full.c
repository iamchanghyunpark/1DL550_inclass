#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main () {
    int N = 512 * 1024 * 1024;
    int *A, *B;
    long dot_product = 0;
    double begin,end;

    A = malloc(sizeof(int) * N);
    B = malloc(sizeof(int) * N);

    if (!A || !B) {
        printf("Error allocating arrays\n");
        exit (1);
    }

    //Initialize arrays
    for (int i = 0; i < N; i++) {
        A[i] = i+1;
        B[i] = N-i;
    }

    //Now running time-stamp
    begin = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        dot_product += A[i] * B[i];
    }
    end = omp_get_wtime();

    printf("The final dotproduct is %ld\n", dot_product);
    printf("Time to execute was %lfs\n", end-begin);

    return 0;
}
