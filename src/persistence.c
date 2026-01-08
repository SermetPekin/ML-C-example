#include "persistence.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MODEL_MAGIC 0x4D4C4E4E
#define MODEL_VERSION 1

// Helper to count parameters in a model
static u32 _count_parameters(const model_context* model) {
    u32 count = 0;
    for (u32 i = 0; i < model->cost_prog.size; i++) {
        model_var* var = model->cost_prog.vars[i];
        if (mv_is_parameter(var)) {
            count++;
        }
    }
    return count;
}

b32 model_save(const model_context* model, const char* filename) {
    if (model == NULL || filename == NULL) {
        fprintf(stderr, "Error: model_save - invalid arguments\n");
        return 0;
    }

    FILE* f = fopen(filename, "wb");
    if (f == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' for writing\n", filename);
        return 0;
    }

    // Count parameters
    u32 num_params = _count_parameters(model);
    if (num_params == 0) {
        fprintf(stderr, "Error: Model has no parameters to save\n");
        fclose(f);
        return 0;
    }

    // Write header
    model_file_header header = {
        .magic = MODEL_MAGIC,
        .version = MODEL_VERSION,
        .num_parameters = num_params,
        .reserved = 0
    };

    if (fwrite(&header, sizeof(model_file_header), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to write header\n");
        fclose(f);
        return 0;
    }

    // Collect parameter metadata
    parameter_metadata* metadata = malloc(num_params * sizeof(parameter_metadata));
    if (metadata == NULL) {
        fprintf(stderr, "Error: Failed to allocate metadata array\n");
        fclose(f);
        return 0;
    }

    u32 param_idx = 0;
    for (u32 i = 0; i < model->cost_prog.size; i++) {
        model_var* var = model->cost_prog.vars[i];
        if (mv_is_parameter(var)) {
            metadata[param_idx].rows = var->val->rows;
            metadata[param_idx].cols = var->val->cols;
            param_idx++;
        }
    }

    // Write metadata
    if (fwrite(metadata, sizeof(parameter_metadata), num_params, f) != num_params) {
        fprintf(stderr, "Error: Failed to write parameter metadata\n");
        free(metadata);
        fclose(f);
        return 0;
    }

    // Write parameter data
    for (u32 i = 0; i < model->cost_prog.size; i++) {
        model_var* var = model->cost_prog.vars[i];
        if (mv_is_parameter(var)) {
            u32 total_elements = var->val->rows * var->val->cols;
            if (fwrite(var->val->data, sizeof(f32), total_elements, f) != total_elements) {
                fprintf(stderr, "Error: Failed to write parameter data\n");
                free(metadata);
                fclose(f);
                return 0;
            }
        }
    }

    free(metadata);
    fclose(f);

    printf("Model saved to '%s' (%u parameters)\n", filename, num_params);
    return 1;
}

b32 model_load(model_context* model, const char* filename) {
    if (model == NULL || filename == NULL) {
        fprintf(stderr, "Error: model_load - invalid arguments\n");
        return 0;
    }

    FILE* f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' for reading\n", filename);
        return 0;
    }

    // Read header
    model_file_header header;
    if (fread(&header, sizeof(model_file_header), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read header\n");
        fclose(f);
        return 0;
    }

    // Validate header
    if (header.magic != MODEL_MAGIC) {
        fprintf(stderr, "Error: Invalid model file (bad magic number)\n");
        fclose(f);
        return 0;
    }

    if (header.version != MODEL_VERSION) {
        fprintf(stderr, "Error: Unsupported model version %u (expected %u)\n",
                header.version, MODEL_VERSION);
        fclose(f);
        return 0;
    }

    // Read metadata
    parameter_metadata* metadata = malloc(header.num_parameters * sizeof(parameter_metadata));
    if (metadata == NULL) {
        fprintf(stderr, "Error: Failed to allocate metadata array\n");
        fclose(f);
        return 0;
    }

    if (fread(metadata, sizeof(parameter_metadata), header.num_parameters, f) != header.num_parameters) {
        fprintf(stderr, "Error: Failed to read parameter metadata\n");
        free(metadata);
        fclose(f);
        return 0;
    }

    // Count model parameters and validate
    u32 model_num_params = _count_parameters(model);
    if (model_num_params != header.num_parameters) {
        fprintf(stderr, "Error: Model parameter count mismatch (file: %u, model: %u)\n",
                header.num_parameters, model_num_params);
        free(metadata);
        fclose(f);
        return 0;
    }

    // Load parameter data
    u32 param_idx = 0;
    for (u32 i = 0; i < model->cost_prog.size && param_idx < header.num_parameters; i++) {
        model_var* var = model->cost_prog.vars[i];
        if (mv_is_parameter(var)) {
            // Validate dimensions
            if (var->val->rows != metadata[param_idx].rows ||
                var->val->cols != metadata[param_idx].cols) {
                fprintf(stderr, "Error: Dimension mismatch for parameter %u\n", param_idx);
                fprintf(stderr, "  File: %u x %u, Model: %u x %u\n",
                        metadata[param_idx].rows, metadata[param_idx].cols,
                        var->val->rows, var->val->cols);
                free(metadata);
                fclose(f);
                return 0;
            }

            // Load data
            u32 total_elements = var->val->rows * var->val->cols;
            if (fread(var->val->data, sizeof(f32), total_elements, f) != total_elements) {
                fprintf(stderr, "Error: Failed to read parameter data for parameter %u\n", param_idx);
                free(metadata);
                fclose(f);
                return 0;
            }

            param_idx++;
        }
    }

    free(metadata);
    fclose(f);

    printf("Model loaded from '%s' (%u parameters)\n", filename, header.num_parameters);
    return 1;
}
