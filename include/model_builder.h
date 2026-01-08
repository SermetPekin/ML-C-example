#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include "model.h"
#include "config_parser.h"
#include "utils/arena.h"

// Build model layers from config and set input/output variables
// Returns true on success, false on error
b32 model_build_from_config(
    mem_arena* arena,
    model_context* model,
    const architecture_config* arch,
    u32 input_size,
    u32 output_size
);

// Create a dense layer: output = activation(W @ input + b)
model_var* model_build_dense_layer(
    mem_arena* arena,
    model_context* model,
    model_var* input,
    u32 output_size,
    activation_type activation
);

// Helper: Create a residual dense layer
// Applies: output = input + activation(W @ input + b)
// Input and output must have same dimension (for residual connection)
// Returns the output variable
model_var* model_build_residual_layer(
    mem_arena* arena,
    model_context* model,
    model_var* input,
    u32 hidden_size,
    activation_type activation
);

#endif // MODEL_BUILDER_H
