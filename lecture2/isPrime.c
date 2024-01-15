#include <stdio.h>
#include <omp.h>

int N = 50000000;

//If is prime, returns 1; else returns 0
int isPrime(int n) {
    //Fill me!
    return 1;
}

int main()
{
    int countPrimes = 0;
    double begin,end;

    //Task: find the total number of prime numbers in the range [2, N]
    begin = omp_get_wtime();
    //Fill me! (Call isPrime)
    end = omp_get_wtime();
    printf("Found %d prime numbers between 1 and %d\n", countPrimes, N);
    printf("Took %lfs to find primes\n", end-begin);
}
