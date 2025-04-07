CC = gcc
CFLAGS = -Wall -Wextra -lm

SOURCES = main.c front_propagation.c back_propagation.c neural_network.c mnist.c
OBJECTS = main.o front_propagation.o back_propagation.o neural_network.o mnist.o

TARGET = main

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(CFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

