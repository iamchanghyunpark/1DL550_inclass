#include "common.h"

double get_time_diff_ms(struct timeval start, struct timeval end)
{
    double sec = end.tv_sec - start.tv_sec;
    double usec = end.tv_usec - start.tv_usec;

    return usec/1000 + sec * 1000;
}
