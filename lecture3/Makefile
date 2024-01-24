CC = gcc
CFLAGS = -Wall -g -O3

TARGET = grayscale_image
SRCS = grayscale_image.c ppm.c common.c

# Detect the architecture
UNAME_S := $(shell uname -m)
ifeq ($(UNAME_S),x86_64)
    CFLAGS += -D__x86_64__
    CFLAGS += -mavx -lm
else ifeq ($(UNAME_S),arm64)
    CC = clang
    CFLAGS += -D__arm__
endif

# Default target
all: $(TARGET) pi

# Rule for building the target
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

pi: pi.c common.c
	$(CC) $(CFLAGS) -o $@ $^


# 'clean' target for removing compiled files
clean:
	rm -f $(TARGET) pi

run_grayscale_x86: grayscale_image
	./grayscale_image serial *.ppm
	./grayscale_image sse *.ppm
	./grayscale_image avx *.ppm

run_grayscale_arm: grayscale_image
	./grayscale_image serial *.ppm
	./grayscale_image neon *.ppm
	./grayscale_image neon_fp16 *.ppm

run_pi_x86: pi
	./pi serial
	./pi sse
	./pi avx

run_pi_arm: pi
	./pi serial
	./pi neon

# Prevent make from doing something with a file named 'all' or 'clean'
.PHONY: all clean
