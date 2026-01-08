#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "config_parser.h"

static void skip_whitespace(const char* line, u32* pos) {
    while (line[*pos] == ' ' || line[*pos] == '\t') {
        (*pos)++;
    }
}

static b32 parse_string(const char* line, const char* key, char* out_value, u32 max_len) {
    u32 pos = 0;
    skip_whitespace(line, &pos);

    u32 key_len = strlen(key);
    if (strncmp(&line[pos], key, key_len) != 0) {
        return 0;
    }

    pos += key_len;
    skip_whitespace(line, &pos);

    if (line[pos] != '=') {
        return 0;
    }

    pos++;
    skip_whitespace(line, &pos);

    u32 out_pos = 0;
    while (line[pos] != '\0' && line[pos] != '\n' && line[pos] != '#' && out_pos < max_len - 1) {
        if (line[pos] == ' ' || line[pos] == '\t') {
            u32 lookahead = pos + 1;
            while (line[lookahead] == ' ' || line[lookahead] == '\t') {
                lookahead++;
            }

            if (line[lookahead] == '\0' || line[lookahead] == '\n' || line[lookahead] == '#') {
                break;
            }
        }

        out_value[out_pos++] = line[pos++];
    }

    out_value[out_pos] = '\0';
    return out_pos > 0;
}

static b32 parse_uint(const char* line, const char* key, u32* out_value) {
    char value_str[64];
    if (!parse_string(line, key, value_str, sizeof(value_str))) {
        return 0;
    }

    char* endptr;
    long val = strtol(value_str, &endptr, 10);
    if (*endptr != '\0' || val < 0) {
        return 0;
    }

    *out_value = (u32)val;
    return 1;
}

static b32 parse_float(const char* line, const char* key, f32* out_value) {
    char value_str[64];
    if (!parse_string(line, key, value_str, sizeof(value_str))) {
        return 0;
    }

    char* endptr;
    float val = strtof(value_str, &endptr);
    if (*endptr != '\0') {
        return 0;
    }

    *out_value = val;
    return 1;
}

static b32 parse_optimizer_type(const char* line, const char* key, optimizer_type* out_optimizer) {
    char value_str[64];
    if (!parse_string(line, key, value_str, sizeof(value_str))) {
        return 0;
    }

    if (strcmp(value_str, "sgd") == 0) {
        *out_optimizer = OPTIMIZER_SGD;
        return 1;
    } else if (strcmp(value_str, "adam") == 0) {
        *out_optimizer = OPTIMIZER_ADAM;
        return 1;
    } else {
        fprintf(stderr, "Error: Unknown optimizer '%s'\n", value_str);
        return 0;
    }
}

static b32 parse_label_format(const char* line, const char* key, label_format_type* out_format) {
    char value_str[64];
    if (!parse_string(line, key, value_str, sizeof(value_str))) {
        return 0;
    }

    if (strcmp(value_str, "integer_indices") == 0) {
        *out_format = LABEL_FORMAT_INTEGER_INDICES;
    } else if (strcmp(value_str, "one_hot") == 0) {
        *out_format = LABEL_FORMAT_ONE_HOT;
    } else if (strcmp(value_str, "raw_floats") == 0) {
        *out_format = LABEL_FORMAT_RAW_FLOATS;
    } else if (strcmp(value_str, "auto") == 0) {
        *out_format = LABEL_FORMAT_AUTO;
    } else {
        fprintf(stderr, "Error: Unknown label format '%s'\n", value_str);
        return 0;
    }

    return 1;
}

static b32 parse_layer_spec(const char* line, layer_spec* out_spec) {
    char value_str[256];
    if (!parse_string(line, "layer", value_str, sizeof(value_str))) {
        return 0;
    }

    // Format: "type input output activation"
    char type_str[32], input_str[32], output_str[32], activation_str[32];
    int matches = sscanf(value_str, "%s %s %s %s", type_str, input_str, output_str, activation_str);

    if (matches != 4) {
        fprintf(stderr, "Error: Invalid layer specification: %s\n", value_str);
        return 0;
    }

    if (strcmp(type_str, "dense") == 0) {
        out_spec->type = LAYER_TYPE_DENSE;
    } else if (strcmp(type_str, "residual_dense") == 0) {
        out_spec->type = LAYER_TYPE_RESIDUAL_DENSE;
    } else {
        fprintf(stderr, "Error: Unknown layer type '%s'\n", type_str);
        return 0;
    }

    char* endptr;
    long input_size = strtol(input_str, &endptr, 10);
    if (*endptr != '\0' || input_size <= 0) {
        fprintf(stderr, "Error: Invalid input_size '%s'\n", input_str);
        return 0;
    }

    long output_size = strtol(output_str, &endptr, 10);
    if (*endptr != '\0' || output_size <= 0) {
        fprintf(stderr, "Error: Invalid output_size '%s'\n", output_str);
        return 0;
    }

    out_spec->input_size = (u32)input_size;
    out_spec->output_size = (u32)output_size;

    if (strcmp(activation_str, "none") == 0) {
        out_spec->activation = ACTIVATION_NONE;
    } else if (strcmp(activation_str, "relu") == 0) {
        out_spec->activation = ACTIVATION_RELU;
    } else if (strcmp(activation_str, "softmax") == 0) {
        out_spec->activation = ACTIVATION_SOFTMAX;
    } else {
        fprintf(stderr, "Error: Unknown activation '%s'\n", activation_str);
        return 0;
    }

    return 1;
}

b32 config_parse(mem_arena* arena, const char* filename, ml_config* out_config) {
    if (out_config == NULL) {
        fprintf(stderr, "Error: out_config is NULL\n");
        return 0;
    }

    memset(out_config, 0, sizeof(ml_config));
    out_config->dataset.label_format = LABEL_FORMAT_AUTO;
    out_config->training.epochs = 10;
    out_config->training.batch_size = 50;
    out_config->training.learning_rate = 0.01f;
    out_config->training.optimizer = OPTIMIZER_SGD;
    out_config->training.adam_beta1 = 0.9f;
    out_config->training.adam_beta2 = 0.999f;
    out_config->training.adam_epsilon = 1e-8f;

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open config file '%s'\n", filename);
        return 0;
    }

    char line[512];
    u32 line_num = 0;
    int current_section = 0;  // 0=none, 1=dataset, 2=training, 3=architecture
    u32 layer_count = 0;

    while (fgets(line, sizeof(line), file) != NULL) {
        line_num++;

        u32 len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
            len--;
        }

        u32 pos = 0;
        skip_whitespace(line, &pos);
        if (line[pos] == '\0' || line[pos] == '#') {
            continue;
        }

        if (line[pos] == '[') {
            if (strncmp(&line[pos], "[dataset]", 9) == 0) {
                current_section = 1;
            } else if (strncmp(&line[pos], "[training]", 10) == 0) {
                current_section = 2;
            } else if (strncmp(&line[pos], "[architecture]", 14) == 0) {
                current_section = 3;
            } else {
                fprintf(stderr, "Warning: Unknown section at line %u: %s\n", line_num, line);
            }

            continue;
        }

        if (current_section == 1) {
            if (parse_string(line, "train_images", out_config->dataset.train_images_path,
                            sizeof(out_config->dataset.train_images_path))) {
                continue;
            }
            if (parse_string(line, "train_labels", out_config->dataset.train_labels_path,
                            sizeof(out_config->dataset.train_labels_path))) {
                continue;
            }
            if (parse_string(line, "test_images", out_config->dataset.test_images_path,
                            sizeof(out_config->dataset.test_images_path))) {
                continue;
            }
            if (parse_string(line, "test_labels", out_config->dataset.test_labels_path,
                            sizeof(out_config->dataset.test_labels_path))) {
                continue;
            }
            if (parse_uint(line, "train_size", &out_config->dataset.train_size)) {
                continue;
            }
            if (parse_uint(line, "test_size", &out_config->dataset.test_size)) {
                continue;
            }
            if (parse_uint(line, "input_size", &out_config->dataset.input_size)) {
                continue;
            }
            if (parse_uint(line, "output_size", &out_config->dataset.output_size)) {
                continue;
            }
            if (parse_label_format(line, "label_format", &out_config->dataset.label_format)) {
                continue;
            }

            fprintf(stderr, "Warning: Unknown dataset key at line %u: %s\n", line_num, line);
        } else if (current_section == 2) {
            if (parse_uint(line, "epochs", &out_config->training.epochs)) {
                continue;
            }
            if (parse_uint(line, "batch_size", &out_config->training.batch_size)) {
                continue;
            }
            if (parse_float(line, "learning_rate", &out_config->training.learning_rate)) {
                continue;
            }
            if (parse_optimizer_type(line, "optimizer", &out_config->training.optimizer)) {
                continue;
            }
            if (parse_float(line, "adam_beta1", &out_config->training.adam_beta1)) {
                continue;
            }
            if (parse_float(line, "adam_beta2", &out_config->training.adam_beta2)) {
                continue;
            }
            if (parse_float(line, "adam_epsilon", &out_config->training.adam_epsilon)) {
                continue;
            }

            fprintf(stderr, "Warning: Unknown training key at line %u: %s\n", line_num, line);
        } else if (current_section == 3) {
            if (strncmp(&line[pos], "layer", 5) == 0) {
                layer_spec spec;
                if (parse_layer_spec(line, &spec)) {
                    // Lazy allocate on first layer (fixed size for simplicity)
                    if (layer_count == 0) {
                        out_config->architecture.layers = PUSH_ARRAY(arena, layer_spec, 50);
                        if (out_config->architecture.layers == NULL) {
                            fprintf(stderr, "Error: Failed to allocate layer array\n");
                            fclose(file);
                            return 0;
                        }
                    }

                    out_config->architecture.layers[layer_count] = spec;
                    layer_count++;

                    if (layer_count >= 50) {
                        fprintf(stderr, "Error: Too many layers (max 50)\n");
                        fclose(file);
                        return 0;
                    }

                    continue;
                }
            }

            fprintf(stderr, "Warning: Unknown architecture line at line %u: %s\n", line_num, line);
        }
    }

    out_config->architecture.num_layers = layer_count;

    fclose(file);

    if (out_config->dataset.train_size == 0) {
        fprintf(stderr, "Error: train_size not specified in config\n");
        return 0;
    }
    if (out_config->dataset.test_size == 0) {
        fprintf(stderr, "Error: test_size not specified in config\n");
        return 0;
    }
    if (out_config->dataset.input_size == 0) {
        fprintf(stderr, "Error: input_size not specified in config\n");
        return 0;
    }
    if (out_config->dataset.output_size == 0) {
        fprintf(stderr, "Error: output_size not specified in config\n");
        return 0;
    }
    if (out_config->architecture.num_layers == 0) {
        fprintf(stderr, "Error: No layers specified in architecture section\n");
        return 0;
    }
    if (strlen(out_config->dataset.train_images_path) == 0) {
        fprintf(stderr, "Error: train_images path not specified\n");
        return 0;
    }
    if (strlen(out_config->dataset.train_labels_path) == 0) {
        fprintf(stderr, "Error: train_labels path not specified\n");
        return 0;
    }
    if (strlen(out_config->dataset.test_images_path) == 0) {
        fprintf(stderr, "Error: test_images path not specified\n");
        return 0;
    }
    if (strlen(out_config->dataset.test_labels_path) == 0) {
        fprintf(stderr, "Error: test_labels path not specified\n");
        return 0;
    }

    return 1;
}

void config_print(const ml_config* config) {
    printf("\n=== Configuration ===\n");

    printf("\n[Dataset]\n");
    printf("  train_images: %s\n", config->dataset.train_images_path);
    printf("  train_labels: %s\n", config->dataset.train_labels_path);
    printf("  test_images:  %s\n", config->dataset.test_images_path);
    printf("  test_labels:  %s\n", config->dataset.test_labels_path);
    printf("  train_size:   %u\n", config->dataset.train_size);
    printf("  test_size:    %u\n", config->dataset.test_size);
    printf("  input_size:   %u\n", config->dataset.input_size);
    printf("  output_size:  %u\n", config->dataset.output_size);

    const char* label_format_str = "unknown";
    switch (config->dataset.label_format) {
        case LABEL_FORMAT_INTEGER_INDICES:
            label_format_str = "integer_indices";
            break;
        case LABEL_FORMAT_ONE_HOT:
            label_format_str = "one_hot";
            break;
        case LABEL_FORMAT_RAW_FLOATS:
            label_format_str = "raw_floats";
            break;
        case LABEL_FORMAT_AUTO:
            label_format_str = "auto";
            break;
    }

    printf("  label_format: %s\n", label_format_str);

    printf("\n[Training]\n");
    printf("  epochs:       %u\n", config->training.epochs);
    printf("  batch_size:   %u\n", config->training.batch_size);
    printf("  learning_rate: %f\n", config->training.learning_rate);

    const char* optimizer_str = (config->training.optimizer == OPTIMIZER_SGD) ? "sgd" : "adam";
    printf("  optimizer:    %s\n", optimizer_str);

    if (config->training.optimizer == OPTIMIZER_ADAM) {
        printf("    beta1:      %f\n", config->training.adam_beta1);
        printf("    beta2:      %f\n", config->training.adam_beta2);
        printf("    epsilon:    %f\n", config->training.adam_epsilon);
    }

    printf("\n[Architecture] (%u layers)\n", config->architecture.num_layers);
    for (u32 i = 0; i < config->architecture.num_layers; i++) {
        const layer_spec* layer = &config->architecture.layers[i];

        const char* type_str = (layer->type == LAYER_TYPE_DENSE) ? "dense" : "residual_dense";
        const char* activation_str = "unknown";

        switch (layer->activation) {
            case ACTIVATION_NONE:
                activation_str = "none";
                break;
            case ACTIVATION_RELU:
                activation_str = "relu";
                break;
            case ACTIVATION_SOFTMAX:
                activation_str = "softmax";
                break;
        }

        printf("  Layer %u: %s %u -> %u (%s)\n",
               i, type_str, layer->input_size, layer->output_size, activation_str);
    }

    printf("\n");
}
