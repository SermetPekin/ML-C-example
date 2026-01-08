#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include "utils/base.h"
#include "utils/arena.h"
#include "label_utils.h"

// Layer specification types
typedef enum {
    LAYER_TYPE_DENSE,
    LAYER_TYPE_RESIDUAL_DENSE
} layer_type;

typedef enum {
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX
} activation_type;

// Configuration structures
typedef struct {
    char train_images_path[256];
    char train_labels_path[256];
    char test_images_path[256];
    char test_labels_path[256];
    u32 train_size;
    u32 test_size;
    u32 input_size;
    u32 output_size;
    label_format_type label_format;
} dataset_config;

typedef struct {
    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} training_config;

typedef struct {
    layer_type type;
    u32 input_size;
    u32 output_size;
    activation_type activation;
} layer_spec;

typedef struct {
    layer_spec* layers;
    u32 num_layers;
} architecture_config;

typedef struct {
    dataset_config dataset;
    training_config training;
    architecture_config architecture;
} ml_config;

// Parse configuration file
// Returns true on success, false on error
b32 config_parse(
    mem_arena* arena,
    const char* filename,
    ml_config* out_config
);

// Print configuration for debugging
void config_print(const ml_config* config);

#endif // CONFIG_PARSER_H
