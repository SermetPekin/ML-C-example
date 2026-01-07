#ifndef MNIST_H
#define MNIST_H

#include "utils/base.h"
#include "utils/arena.h"
#include "model.h"

// Visualization
void draw_mnist_digit(f32* data);

// Model creation
void create_mnist_model(mem_arena* arena, model_context* model);

#endif // MNIST_H
