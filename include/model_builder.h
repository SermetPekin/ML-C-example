#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include "model.h"
#include "config_parser.h"
#include "utils/arena.h"

// Build layers from architecture config and set input/output nodes
b32 model_build_from_config(
    mem_arena* arena,
    model_context* model,
    const architecture_config* arch,
    u32 input_size,
    u32 output_size
);

// Dense layer with weights, bias, and activation
model_var* model_build_dense_layer(
    mem_arena* arena,
    model_context* model,
    model_var* input,
    u32 output_size,
    activation_type activation
);

// Residual layer with skip connection (requires matching input/output dims)
model_var* model_build_residual_layer(
    mem_arena* arena,
    model_context* model,
    model_var* input,
    u32 hidden_size,
    activation_type activation
);

#endif // MODEL_BUILDER_H
