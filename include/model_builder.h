#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include "model.h"
#include "config_parser.h"
#include "utils/arena.h"

// Build model from architecture specification
// Automatically sets model->input, model->output, model->desired_output, model->cost
// Returns true on success, false on error
b32 model_build_from_config(
    mem_arena* arena,
    model_context* model,
    const architecture_config* arch,
    u32 input_size,
    u32 output_size
);

// Helper: Create a dense layer
// Applies: z = W @ input + b, then activation(z)
// Returns the output variable
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
