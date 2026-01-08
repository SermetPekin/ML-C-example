#include <stdio.h>
#include <math.h>
#include "model_builder.h"
#include "utils/prng.h"

model_var* model_build_dense_layer(
    mem_arena* arena,
    model_context* model,
    model_var* input,
    u32 output_size,
    activation_type activation
) {
    if (input == NULL || input->val == NULL) {
        fprintf(stderr, "Error: Invalid input to dense layer\n");
        return NULL;
    }

    u32 input_size = input->val->rows;

    // Create weight matrix W (output_size x input_size)
    model_var* W = mv_create(arena, model, output_size, input_size,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    if (W == NULL) {
        fprintf(stderr, "Error: Failed to create weight matrix\n");
        return NULL;
    }

    // Create bias vector b (output_size x 1)
    model_var* b = mv_create(arena, model, output_size, 1,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    if (b == NULL) {
        fprintf(stderr, "Error: Failed to create bias vector\n");
        return NULL;
    }

    // Xavier initialization: uniform(-sqrt(6/(in+out)), sqrt(6/(in+out)))
    f32 bound = sqrtf(6.0f / (input_size + output_size));
    mat_fill_rand(W->val, -bound, bound);
    mat_fill_rand(b->val, -bound, bound);

    // Build computation graph: z = W @ input + b
    model_var* z_matmul = mv_matmul(arena, model, W, input, 0);
    if (z_matmul == NULL) {
        fprintf(stderr, "Error: Failed to create matmul operation\n");
        return NULL;
    }

    model_var* z = mv_add(arena, model, z_matmul, b, 0);
    if (z == NULL) {
        fprintf(stderr, "Error: Failed to create add operation\n");
        return NULL;
    }

    // Apply activation function
    model_var* output = NULL;

    if (activation == ACTIVATION_RELU) {
        output = mv_relu(arena, model, z, 0);
    } else if (activation == ACTIVATION_SOFTMAX) {
        output = mv_softmax(arena, model, z, 0);
    } else if (activation == ACTIVATION_NONE) {
        output = z;  // No activation
    } else {
        fprintf(stderr, "Error: Unknown activation type %d\n", activation);
        return NULL;
    }

    return output;
}

model_var* model_build_residual_layer(
    mem_arena* arena,
    model_context* model,
    model_var* input,
    u32 hidden_size,
    activation_type activation
) {
    if (input == NULL || input->val == NULL) {
        fprintf(stderr, "Error: Invalid input to residual layer\n");
        return NULL;
    }

    u32 input_size = input->val->rows;

    // For residual connection, input and output must have same dimension
    if (input_size != hidden_size) {
        fprintf(stderr,
            "Error: Residual layer requires matching input/hidden sizes (%u != %u)\n",
            input_size, hidden_size);
        return NULL;
    }

    // Create weight matrix W (hidden_size x input_size)
    model_var* W = mv_create(arena, model, hidden_size, input_size,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    if (W == NULL) {
        fprintf(stderr, "Error: Failed to create weight matrix for residual layer\n");
        return NULL;
    }

    // Create bias vector b (hidden_size x 1)
    model_var* b = mv_create(arena, model, hidden_size, 1,
                            MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    if (b == NULL) {
        fprintf(stderr, "Error: Failed to create bias vector for residual layer\n");
        return NULL;
    }

    // Xavier initialization
    f32 bound = sqrtf(6.0f / (input_size + hidden_size));
    mat_fill_rand(W->val, -bound, bound);
    mat_fill_rand(b->val, -bound, bound);

    // Build computation graph: z = W @ input + b
    model_var* z_matmul = mv_matmul(arena, model, W, input, 0);
    if (z_matmul == NULL) {
        fprintf(stderr, "Error: Failed to create matmul in residual layer\n");
        return NULL;
    }

    model_var* z_add = mv_add(arena, model, z_matmul, b, 0);
    if (z_add == NULL) {
        fprintf(stderr, "Error: Failed to create add in residual layer\n");
        return NULL;
    }

    // Apply activation
    model_var* activated = NULL;

    if (activation == ACTIVATION_RELU) {
        activated = mv_relu(arena, model, z_add, 0);
    } else if (activation == ACTIVATION_SOFTMAX) {
        activated = mv_softmax(arena, model, z_add, 0);
    } else if (activation == ACTIVATION_NONE) {
        activated = z_add;
    } else {
        fprintf(stderr, "Error: Unknown activation type %d in residual layer\n", activation);
        return NULL;
    }

    // Add residual connection: output = input + activated
    model_var* output = mv_add(arena, model, input, activated, 0);
    if (output == NULL) {
        fprintf(stderr, "Error: Failed to create residual connection\n");
        return NULL;
    }

    return output;
}

b32 model_build_from_config(
    mem_arena* arena,
    model_context* model,
    const architecture_config* arch,
    u32 input_size,
    u32 output_size
) {
    if (model == NULL || arch == NULL) {
        fprintf(stderr, "Error: Invalid model or architecture config\n");
        return 0;
    }

    if (arch->num_layers == 0) {
        fprintf(stderr, "Error: Architecture has no layers\n");
        return 0;
    }

    // Create input node
    model_var* input = mv_create(arena, model, input_size, 1, MV_FLAG_INPUT);
    if (input == NULL) {
        fprintf(stderr, "Error: Failed to create input node\n");
        return 0;
    }

    model->input = input;

    // Build layers
    model_var* current = input;

    for (u32 i = 0; i < arch->num_layers; i++) {
        const layer_spec* layer = &arch->layers[i];

        // Verify layer input size matches current output
        u32 expected_input_size = (i == 0) ? input_size : arch->layers[i - 1].output_size;

        if (layer->input_size != expected_input_size) {
            fprintf(stderr,
                "Error: Layer %u has input_size %u but expected %u\n",
                i, layer->input_size, expected_input_size);
            return 0;
        }

        model_var* layer_output = NULL;

        if (layer->type == LAYER_TYPE_DENSE) {
            layer_output = model_build_dense_layer(arena, model, current,
                                                   layer->output_size, layer->activation);
        } else if (layer->type == LAYER_TYPE_RESIDUAL_DENSE) {
            layer_output = model_build_residual_layer(arena, model, current,
                                                      layer->output_size, layer->activation);
        } else {
            fprintf(stderr, "Error: Unknown layer type %d\n", layer->type);
            return 0;
        }

        if (layer_output == NULL) {
            fprintf(stderr, "Error: Failed to build layer %u\n", i);
            return 0;
        }

        current = layer_output;
    }

    // Verify final layer output matches expected output size
    if (arch->layers[arch->num_layers - 1].output_size != output_size) {
        fprintf(stderr,
            "Error: Final layer output %u does not match expected output_size %u\n",
            arch->layers[arch->num_layers - 1].output_size, output_size);
        return 0;
    }

    // Set output node (mark with flag)
    model->output = mv_create(arena, model, output_size, 1, MV_FLAG_OUTPUT);
    if (model->output == NULL) {
        fprintf(stderr, "Error: Failed to create output node\n");
        return 0;
    }

    // The actual output comes from the last layer
    // We'll use current as the prediction output
    model->output->val = current->val;  // Share the value matrix

    // Create desired output node for labels
    model_var* y = mv_create(arena, model, output_size, 1, MV_FLAG_DESIRED_OUTPUT);
    if (y == NULL) {
        fprintf(stderr, "Error: Failed to create desired output node\n");
        return 0;
    }

    model->desired_output = y;

    // Create cost node (cross-entropy loss)
    model_var* cost = mv_cross_entropy(arena, model, y, current, MV_FLAG_COST);
    if (cost == NULL) {
        fprintf(stderr, "Error: Failed to create cost node\n");
        return 0;
    }

    model->cost = cost;

    return 1;
}
