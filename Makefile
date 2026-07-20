# Compiler and Flags
CC = gcc
CFLAGS = -O3 -Wall -Wextra
LDFLAGS = -lm

# Files
SRCS = main.c loader.c math.c
OBJS = $(SRCS:.c=.o)
TARGET = engine

# Build Rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)