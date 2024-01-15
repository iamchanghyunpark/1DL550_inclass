#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main () {
    int N = 512 * 1024 * 1024;
    int *A, *B;
    long dot_product = 0;
    double begin,end; // Timestamps for beginning and end of computation

    //Allocate Array A and B with N elements
    //Fill in

    //Initialize arrays
    //Fill in. Loop through A and B and fill in initial value

    //Now running time-stamp
    begin = omp_get_wtime();
    //Fill in. Loop and generate dot-product
    end = omp_get_wtime();

    printf("The final dotproduct is %ld\n", dot_product);
    printf("Time to execute was %lfs\n", end-begin);

    return 0;
}
