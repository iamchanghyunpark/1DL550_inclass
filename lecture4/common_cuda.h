#ifndef COMMON_H
#define COMMON_H

#include <sys/time.h>

#define MALLOC_ALIGN 32
#define TILE_DIM 64
#define BLOCK_DIM 16

extern "C" double get_time_diff_ms(struct timeval start, struct timeval end);

#endif /* COMMON_H */
