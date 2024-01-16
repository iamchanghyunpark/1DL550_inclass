#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>

int N = 512 * 1024 * 1024;
int *A, *B;
int K = 4; //How many threads we will spawn

void * dot_product_func (void *arg) {
    long dot_product = 0;
    int thread_id = (int)arg;
    int offset = thread_id * N/K;
    for (int i = 0; i < N/K; i++) {
        dot_product += A[i+offset] * B[i+offset];
    }
    return (void*)dot_product;
}

int main () {
    int i;
    long dot_product = 0;
    double begin,end;

    A = malloc(sizeof(int) * N);
    B = malloc(sizeof(int) * N);

    if (!A || !B) {
        printf("Error allocating arrays\n");
        exit (1);
    }

    //Initialize arrays
    for (i = 0; i < N; i++) {
        A[i] = i+1;
        B[i] = N-i;
    }

    //Now running time-stamp
    begin = omp_get_wtime();
    pthread_t threads[K];
    for (int n = 0; n < K; n++) {
        pthread_create(&threads[n], NULL, dot_product_func, (void*)n);
    }
    for (int n = 0; n < K; n++) {
        long ret;
        pthread_join(threads[n], (void **)&ret);
        dot_product += ret;
    }
    end = omp_get_wtime();

    printf("The final dotproduct is %ld\n", dot_product);
    printf("Time to execute was %lfs\n", end-begin);

    return 0;
}
