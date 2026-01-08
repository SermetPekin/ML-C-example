#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>

#include "config_parser.h"
#include "model_builder.h"
#include "label_utils.h"
#include "model.h"
#include "matrix.h"
#include "utils/arena.h"

int main(int argc, char** argv) {
    // Parse command-line arguments
    const char* config_file = (argc > 1) ? argv[1] : "model_config.txt";

    // Initialize memory arena
    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));
    if (perm_arena == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory arena.\n");
        return 1;
    }

    // Parse configuration
    ml_config config;
    if (!config_parse(perm_arena, config_file, &config)) {
        fprintf(stderr, "Error: Failed to parse config file '%s'\n", config_file);
        return 1;
    }

    config_print(&config);

    // Load training images
    matrix* train_images = mat_load(perm_arena,
        config.dataset.train_size,
        config.dataset.input_size,
        config.dataset.train_images_path);

    if (train_images == NULL) {
        fprintf(stderr, "Error: Failed to load training images from '%s'\n",
                config.dataset.train_images_path);
        return 1;
    }

    // Load test images
    matrix* test_images = mat_load(perm_arena,
        config.dataset.test_size,
        config.dataset.input_size,
        config.dataset.test_images_path);

    if (test_images == NULL) {
        fprintf(stderr, "Error: Failed to load test images from '%s'\n",
                config.dataset.test_images_path);
        return 1;
    }

    // Load raw label data
    // Start by loading as single column (works for integer indices)
    matrix* train_labels_raw = mat_load(perm_arena,
        config.dataset.train_size,
        1,
        config.dataset.train_labels_path);

    matrix* test_labels_raw = mat_load(perm_arena,
        config.dataset.test_size,
        1,
        config.dataset.test_labels_path);

    if (train_labels_raw == NULL || test_labels_raw == NULL) {
        fprintf(stderr, "Error: Failed to load label files\n");
        return 1;
    }

    // Detect label format if auto
    label_format_type actual_format = config.dataset.label_format;
    if (actual_format == LABEL_FORMAT_AUTO) {
        actual_format = label_detect_format(train_labels_raw, config.dataset.output_size);
        printf("Auto-detected label format: %d\n", actual_format);
    }

    // Convert labels to one-hot format
    matrix* train_labels = label_convert_to_onehot(perm_arena,
        train_labels_raw,
        actual_format,
        config.dataset.output_size);

    matrix* test_labels = label_convert_to_onehot(perm_arena,
        test_labels_raw,
        actual_format,
        config.dataset.output_size);

    if (train_labels == NULL || test_labels == NULL) {
        fprintf(stderr, "Error: Failed to convert labels to one-hot format\n");
        return 1;
    }

    // Validate labels
    if (!label_validate(train_labels, LABEL_FORMAT_ONE_HOT, config.dataset.output_size)) {
        fprintf(stderr, "Error: Invalid training labels\n");
        return 1;
    }

    if (!label_validate(test_labels, LABEL_FORMAT_ONE_HOT, config.dataset.output_size)) {
        fprintf(stderr, "Error: Invalid test labels\n");
        return 1;
    }

    // Create model context
    model_context* model = model_create(perm_arena);
    if (model == NULL) {
        fprintf(stderr, "Error: Failed to create model context.\n");
        return 1;
    }

    // Build model from architecture config
    if (!model_build_from_config(perm_arena, model, &config.architecture,
                                 config.dataset.input_size,
                                 config.dataset.output_size)) {
        fprintf(stderr, "Error: Failed to build model from architecture config\n");
        return 1;
    }

    // Validate model structure
    if (model->input == NULL || model->output == NULL || model->cost == NULL) {
        fprintf(stderr, "Error: Model structure is incomplete.\n");
        return 1;
    }

    // Compile model
    model_compile(perm_arena, model);

    // Pre-training feedforward on first test sample
    memcpy(model->input->val->data, test_images->data,
           sizeof(f32) * config.dataset.input_size);

    if (!model_feedforward(model)) {
        fprintf(stderr, "Error: Pre-training feedforward failed.\n");
        return 1;
    }

    printf("Pre-training output: ");
    for (u32 i = 0; i < config.dataset.output_size; i++) {
        printf("%.2f ", model->output->val->data[i]);
    }
    printf("\n");

    // Train the model
    model_training_desc training_desc = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,
        .epochs = config.training.epochs,
        .batch_size = config.training.batch_size,
        .learning_rate = config.training.learning_rate
    };

    model_train(model, &training_desc);

    // Post-training feedforward on first test sample
    memcpy(model->input->val->data, test_images->data,
           sizeof(f32) * config.dataset.input_size);

    if (!model_feedforward(model)) {
        fprintf(stderr, "Error: Post-training feedforward failed.\n");
        return 1;
    }

    printf("Post-training output: ");
    for (u32 i = 0; i < config.dataset.output_size; i++) {
        printf("%f ", model->output->val->data[i]);
    }
    printf("\n\n");

    // Cleanup
    arena_destroy(perm_arena);

    return 0;
}
