#define _CRT_SECURE_NO_WARNINGS

#include <inttypes.h>
#include <stdio.h>

#include "matrix.h"
#include "model.h"
#include "mnist.h"
#include "config.h"
#include "../include/utils/arena.h"


int main(void) {
    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));
    if (perm_arena == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory arena.\n");
        return 1;
    }

    matrix* train_images = mat_load(perm_arena, MNIST_TRAIN_SIZE, MNIST_INPUT_SIZE, "train_images.mat");
    matrix* test_images = mat_load(perm_arena, MNIST_TEST_SIZE, MNIST_INPUT_SIZE, "test_images.mat");

    if (train_images == NULL || test_images == NULL) {
        fprintf(stderr, "Error: Data files not found. Please run 'python mnist.py' first to generate the data files.\n");
        return 1;
    }

    matrix* train_labels = mat_create(perm_arena, MNIST_TRAIN_SIZE, MNIST_NUM_CLASSES);
    matrix* test_labels = mat_create(perm_arena, MNIST_TEST_SIZE, MNIST_NUM_CLASSES);

    {
        matrix* train_labels_file = mat_load(perm_arena, MNIST_TRAIN_SIZE, MNIST_LABEL_DIMS, "train_labels.mat");
        matrix* test_labels_file = mat_load(perm_arena, MNIST_TEST_SIZE, MNIST_LABEL_DIMS, "test_labels.mat");

        if (train_labels_file == NULL || test_labels_file == NULL) {
            fprintf(stderr, "Error: Label files not found. Please run 'python mnist.py' first to generate the data files.\n");
            return 1;
        }

        for (u32 i = 0; i < MNIST_TRAIN_SIZE; i++) {
            u32 num = (u32)train_labels_file->data[i];
            if (num >= MNIST_NUM_CLASSES) {
                fprintf(stderr, "Error: Invalid label value %u in train_labels.mat at index %u. Expected value < %u.\n",
                        num, i, MNIST_NUM_CLASSES);
                return 1;
            }
            train_labels->data[i * MNIST_NUM_CLASSES + num] = 1.0f;
        }

        for (u32 i = 0; i < MNIST_TEST_SIZE; i++) {
            u32 num = (u32)test_labels_file->data[i];
            if (num >= MNIST_NUM_CLASSES) {
                fprintf(stderr, "Error: Invalid label value %u in test_labels.mat at index %u. Expected value < %u.\n",
                        num, i, MNIST_NUM_CLASSES);
                return 1;
            }
            test_labels->data[i * MNIST_NUM_CLASSES + num] = 1.0f;
        }
    }

    draw_mnist_digit(test_images->data);
    for (u32 i = 0; i < MNIST_NUM_CLASSES; i++) {
        printf("%.0f ", test_labels->data[i]);
    }
    printf("\n\n");

    model_context* model = model_create(perm_arena);
    if (model == NULL) {
        fprintf(stderr, "Error: Failed to create model context.\n");
        return 1;
    }

    create_mnist_model(perm_arena, model);

    // Validate model structure
    if (model->input == NULL || model->output == NULL || model->cost == NULL) {
        fprintf(stderr, "Error: Model structure is incomplete. Missing input, output, or cost node.\n");
        return 1;
    }

    model_compile(perm_arena, model);

    memcpy(model->input->val->data, test_images->data, sizeof(f32) * MNIST_INPUT_SIZE);
    if (!model_feedforward(model)) {
        fprintf(stderr, "Error: Failed to compute pre-training output.\n");
        return 1;
    }

    printf("pre-training output: ");
    for (u32 i = 0; i < MNIST_NUM_CLASSES; i++) {
        printf("%.2f ", model->output->val->data[i]);
    }
    printf("\n");

    model_training_desc training_desc = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,

        .epochs = DEFAULT_EPOCHS,
        .batch_size = DEFAULT_BATCH_SIZE,
        .learning_rate = DEFAULT_LEARNING_RATE
    };
    model_train(model, &training_desc);

    memcpy(model->input->val->data, test_images->data, sizeof(f32) * MNIST_INPUT_SIZE);
    if (!model_feedforward(model)) {
        fprintf(stderr, "Error: Failed to compute post-training output.\n");
        return 1;
    }
    printf("post-training output: ");
    for (u32 i = 0; i < MNIST_NUM_CLASSES; i++) {
        printf("%f ", model->output->val->data[i]);
    }
    printf("\n\n");

    arena_destroy(perm_arena);

    return 0;
}
