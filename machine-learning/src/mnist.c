#include <stdio.h>
#include <math.h>

#include "mnist.h"
#include "model.h"
#include "matrix.h"
#include "../include/config.h"
#include "utils/arena.h"

void draw_mnist_digit(f32* data) {
    for (u32 y = 0; y < MNIST_PIXEL_WIDTH; y++) {
        for (u32 x = 0; x < MNIST_PIXEL_WIDTH; x++) {
            f32 num = data[x + y * MNIST_PIXEL_WIDTH];
            u32 col = 232 + (u32)(num * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

void create_mnist_model(mem_arena* arena, model_context* model) {
    model_var* input = mv_create(arena, model, MNIST_INPUT_SIZE, 1, MV_FLAG_INPUT);

    model_var* W0 = mv_create(arena, model, MODEL_HIDDEN_SIZE, MNIST_INPUT_SIZE, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W1 = mv_create(arena, model, MODEL_HIDDEN_SIZE, MODEL_HIDDEN_SIZE, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* W2 = mv_create(arena, model, MNIST_NUM_CLASSES, MODEL_HIDDEN_SIZE, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    f32 bound0 = sqrtf(6.0f / (MNIST_INPUT_SIZE + MODEL_HIDDEN_SIZE));
    f32 bound1 = sqrtf(6.0f / (MODEL_HIDDEN_SIZE + MODEL_HIDDEN_SIZE));
    f32 bound2 = sqrtf(6.0f / (MODEL_HIDDEN_SIZE + MNIST_NUM_CLASSES));
    mat_fill_rand(W0->val, -bound0, bound0);
    mat_fill_rand(W1->val, -bound1, bound1);
    mat_fill_rand(W2->val, -bound2, bound2);

    model_var* b0 = mv_create(arena, model, MODEL_HIDDEN_SIZE, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b1 = mv_create(arena, model, MODEL_HIDDEN_SIZE, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    model_var* b2 = mv_create(arena, model, MNIST_NUM_CLASSES, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    model_var* z0_a = mv_matmul(arena, model, W0, input, 0);
    model_var* z0_b = mv_add(arena, model, z0_a, b0, 0);
    model_var* a0 = mv_relu(arena, model, z0_b, 0);

    model_var* z1_a = mv_matmul(arena, model, W1, a0, 0);
    model_var* z1_b = mv_add(arena, model, z1_a, b1, 0);
    model_var* z1_c = mv_relu(arena, model, z1_b, 0);
    model_var* a1 = mv_add(arena, model, a0, z1_c, 0);

    model_var* z2_a = mv_matmul(arena, model, W2, a1, 0);
    model_var* z2_b = mv_add(arena, model, z2_a, b2, 0);
    model_var* output = mv_softmax(arena, model, z2_b, MV_FLAG_OUTPUT);

    model_var* y = mv_create(arena, model, MNIST_NUM_CLASSES, 1, MV_FLAG_DESIRED_OUTPUT);

    // Cost node is created automatically via MV_FLAG_COST
    (void)mv_cross_entropy(arena, model, y, output, MV_FLAG_COST);
}
