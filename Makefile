CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -std=c11
LDFLAGS ?= -fopenmp -lm

TARGET = dpuf-cli
LIBTARGET = libdpuf_avx2.so

OBJS = dpuf_avx2.o dpuf_engine.o dpuf_vision.o dpuf-cli.o

all: $(TARGET) $(LIBTARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(LIBTARGET): dpuf_avx2.o dpuf_engine.o dpuf_vision.o
	$(CC) -shared -fPIC $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c dpuf_kernels.h
	$(CC) -fPIC $(CFLAGS) -c $< -o $@

dpuf_vision.o: dpuf_vision.c dpuf_vision.h dpuf_kernels.h stb_image.h
	$(CC) -fPIC $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET) $(LIBTARGET)

.PHONY: all clean
