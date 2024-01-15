CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g

TARGET = serial pthread openmp isPrime

all: $(TARGET)

serial: serial.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< -fopenmp

pthread: pthread.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< -fopenmp -pthread

openmp: openmp.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< -fopenmp

isPrime: isPrime.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< -fopenmp


clean:
	rm -f $(TARGET)

.PHONY: all clean
