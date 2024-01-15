#include <stdio.h>
#include <omp.h>

double step;
static long num_steps = 100000000L;

int main (){
    long i;
    double x, pi = 0.0, sum=0.0;
    step = 1.0/(double) num_steps; // step number of very small bars!
    double T1, T2;

    T1 = omp_get_wtime();
    //FILL ME
    //For each small bar (x = 0 to num_steps) calculate the y value of this small bar.
    //Then multiply the y values with the total number of steps to get the area (integral)

    T2 = omp_get_wtime();

    printf("Time: %lf, pi: %lf\n", T2-T1, pi);
    return 0;
}
