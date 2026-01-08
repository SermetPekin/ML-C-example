CC := clang
CFLAGS := -Wall -Wextra -std=c99 -O2 -I./include

ifeq ($(OS),Windows_NT)
    LDFLAGS :=
else
    LDFLAGS := -lm
endif

# Directories
SRC_DIR := src
OBJ_DIR := build
BIN_DIR := build
TEST_DIR := tests
TEST_BUILD_DIR := $(BIN_DIR)/tests

# Source files
SOURCES := $(SRC_DIR)/main.c $(SRC_DIR)/matrix.c $(SRC_DIR)/model.c \
           $(SRC_DIR)/config_parser.c $(SRC_DIR)/model_builder.c $(SRC_DIR)/label_utils.c \
           $(SRC_DIR)/persistence.c $(SRC_DIR)/utils/arena.c $(SRC_DIR)/utils/prng.c
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET := $(BIN_DIR)/ml_trainer

# Test files
TEST_SOURCES := $(SRC_DIR)/matrix.c $(SRC_DIR)/model.c \
                $(SRC_DIR)/config_parser.c $(SRC_DIR)/model_builder.c $(SRC_DIR)/label_utils.c \
                $(SRC_DIR)/persistence.c $(SRC_DIR)/utils/arena.c $(SRC_DIR)/utils/prng.c
TEST_OBJECTS := $(TEST_SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TEST_MATRIX_OBJECTS := $(TEST_OBJECTS) $(OBJ_DIR)/utils/arena.o $(OBJ_DIR)/utils/prng.o
TEST_ARENA_OBJECTS := $(OBJ_DIR)/utils/arena.o
TEST_PRNG_OBJECTS := $(OBJ_DIR)/utils/prng.o

# Default target
all: $(TARGET)

# Create build directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR) $(OBJ_DIR)/utils

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/utils/%.o: $(SRC_DIR)/utils/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS)
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Run the application
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Generate MNIST data
data:
	python mnist.py

# Test targets
test: test_matrix test_arena test_prng test_persistence

test_matrix: $(TEST_BUILD_DIR)/test_matrix
	./$(TEST_BUILD_DIR)/test_matrix

test_arena: $(TEST_BUILD_DIR)/test_arena
	./$(TEST_BUILD_DIR)/test_arena

test_prng: $(TEST_BUILD_DIR)/test_prng
	./$(TEST_BUILD_DIR)/test_prng

test_persistence: $(TEST_BUILD_DIR)/test_persistence
	./$(TEST_BUILD_DIR)/test_persistence

$(TEST_BUILD_DIR):
	mkdir -p $(TEST_BUILD_DIR)

$(TEST_BUILD_DIR)/test_matrix: $(TEST_DIR)/test_matrix.c $(OBJ_DIR)/matrix.o $(OBJ_DIR)/utils/arena.o $(OBJ_DIR)/utils/prng.o | $(TEST_BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TEST_BUILD_DIR)/test_arena: $(TEST_DIR)/test_arena.c $(OBJ_DIR)/utils/arena.o | $(TEST_BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TEST_BUILD_DIR)/test_prng: $(TEST_DIR)/test_prng.c $(OBJ_DIR)/utils/prng.o | $(TEST_BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TEST_BUILD_DIR)/test_persistence: $(TEST_DIR)/test_persistence.c $(OBJ_DIR)/matrix.o $(OBJ_DIR)/model.o $(OBJ_DIR)/persistence.o $(OBJ_DIR)/utils/arena.o $(OBJ_DIR)/utils/prng.o | $(TEST_BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Phony targets
.PHONY: all run clean data test test_matrix test_arena test_prng test_persistence
