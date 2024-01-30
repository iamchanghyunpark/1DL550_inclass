TARGETS = grayscale_image pi matmult
SOURCES = $(shell echo *.cu) $(shell echo *.c)
OBJECTS = $(SOURCES:.cu=.o)
CC = nvcc
CFLAGS = -O2 -g -arch=sm_50 -lcudart -Wno-deprecated-gpu-targets

.PHONY: all clean

all: $(TARGETS)

grayscale_image : grayscale_image.cu common.c ppm.c
	$(CC) $(CFLAGS) $^  -o $@

pi: pi.cu common.c
	$(CC) $(CFLAGS) $^  -o $@

matmult: matmult.cu
	$(CC) $(CFLAGS) $^  -o $@



clean:
	-rm $(TARGETS)

