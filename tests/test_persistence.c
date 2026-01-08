#include <stdio.h>
#include <stdlib.h>
#include "utils/base.h"
#include "utils/arena.h"
#include "matrix.h"
#include "model.h"
#include "persistence.h"
#include "test_utils.h"

void create_simple_model(mem_arena* arena, model_context* model) {
    // Create a simple 2-layer network: 4 -> 3 -> 2

    // Input
    model->input = mv_create(arena, model, 4, 1, MV_FLAG_INPUT);

    // Layer 1: 4 -> 3
    model_var* W1 = mv_create(arena, model, 3, 4, MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD);
    model_var* b1 = mv_create(arena, model, 3, 1, MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD);
    mat_fill_rand(W1->val, -0.1f, 0.1f);
    mat_fill(b1->val, 0.0f);

    model_var* z1 = mv_matmul(arena, model, W1, model->input, 0);
    model_var* a1 = mv_add(arena, model, z1, b1, 0);
    model_var* h1 = mv_relu(arena, model, a1, MV_FLAG_REQUIRES_GRAD);

    // Layer 2: 3 -> 2
    model_var* W2 = mv_create(arena, model, 2, 3, MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD);
    model_var* b2 = mv_create(arena, model, 2, 1, MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD);
    mat_fill_rand(W2->val, -0.1f, 0.1f);
    mat_fill(b2->val, 0.0f);

    model_var* z2 = mv_matmul(arena, model, W2, h1, 0);
    model_var* output = mv_add(arena, model, z2, b2, MV_FLAG_OUTPUT | MV_FLAG_REQUIRES_GRAD);

    // Loss (just for structure, won't use it in this test)
    model_var* desired_output = mv_create(arena, model, 2, 1, MV_FLAG_DESIRED_OUTPUT);
    model_var* cost = mv_cross_entropy(arena, model, desired_output, output, MV_FLAG_COST);

    model->output = output;
    model->desired_output = desired_output;
    model->cost = cost;
}

int main(void) {
    TEST_SUITE("Model Persistence Tests");

    // Create arena for test allocations
    mem_arena* arena = arena_create(256 * 1024 * 1024, 8 * 1024 * 1024);

    const char* test_file = "test_model_temp.bin";

    // Test 1: Create model, save it, load it, and verify weights match
    {
        model_context* model1 = model_create(arena);
        create_simple_model(arena, model1);
        model_compile(arena, model1);

        // Store original weights
        f32 original_w1_0 = model1->cost_prog.vars[1]->val->data[0];  // First W1 value

        // Save model
        b32 save_result = model_save(model1, test_file);
        ASSERT_TRUE(save_result);
        TEST("Model save succeeded");

        // Verify file exists and is not empty
        FILE* f = fopen(test_file, "rb");
        ASSERT_TRUE(f != NULL);
        TEST("Model file was created");

        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        ASSERT_TRUE(file_size > 0);
        TEST("Model file is not empty");
        fclose(f);

        // Create a new model with same structure
        model_context* model2 = model_create(arena);
        create_simple_model(arena, model2);
        model_compile(arena, model2);

        // Modify weights to ensure they're different before loading
        mat_fill(model2->cost_prog.vars[1]->val, 0.5f);  // W1

        f32 modified_w1_0 = model2->cost_prog.vars[1]->val->data[0];
        // Ensure they're different
        ASSERT_TRUE(modified_w1_0 != original_w1_0);
        TEST("Weights were different before loading");

        // Load weights
        b32 load_result = model_load(model2, test_file);
        ASSERT_TRUE(load_result);
        TEST("Model load succeeded");

        // Verify weights match
        f32 loaded_w1_0 = model2->cost_prog.vars[1]->val->data[0];
        ASSERT_FLOAT_EQ(loaded_w1_0, original_w1_0, 1e-6f);
        TEST("Loaded weights match saved weights");
    }

    // Test 2: Error handling - try to load non-existent file
    {
        model_context* model = model_create(arena);
        create_simple_model(arena, model);
        model_compile(arena, model);

        b32 result = model_load(model, "nonexistent_model_file.bin");
        ASSERT_TRUE(result == 0);
        TEST("Load fails gracefully for non-existent file");
    }

    // Test 3: Error handling - try to load with mismatched dimensions
    {
        // Create and save a model
        model_context* model1 = model_create(arena);
        create_simple_model(arena, model1);
        model_compile(arena, model1);
        model_save(model1, test_file);

        // Create a different model structure
        model_context* model2 = model_create(arena);
        model2->input = mv_create(arena, model2, 5, 1, MV_FLAG_INPUT);  // Different input size!

        model_var* W1 = mv_create(arena, model2, 3, 5, MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD);
        model_var* b1 = mv_create(arena, model2, 3, 1, MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD);

        model_var* z1 = mv_matmul(arena, model2, W1, model2->input, 0);
        model_var* a1 = mv_add(arena, model2, z1, b1, 0);
        model_var* h1 = mv_relu(arena, model2, a1, MV_FLAG_REQUIRES_GRAD);

        model_var* W2 = mv_create(arena, model2, 2, 3, MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD);
        model_var* b2 = mv_create(arena, model2, 2, 1, MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD);

        model_var* z2 = mv_matmul(arena, model2, W2, h1, 0);
        model_var* output = mv_add(arena, model2, z2, b2, MV_FLAG_OUTPUT | MV_FLAG_REQUIRES_GRAD);

        model_var* desired = mv_create(arena, model2, 2, 1, MV_FLAG_DESIRED_OUTPUT);
        mv_cross_entropy(arena, model2, desired, output, MV_FLAG_COST);

        model2->output = output;
        model2->desired_output = desired;
        model_compile(arena, model2);

        // Try to load - should fail due to dimension mismatch
        b32 result = model_load(model2, test_file);
        ASSERT_TRUE(result == 0);
        TEST("Load fails gracefully for dimension mismatch");
    }

    // Clean up test file
    remove(test_file);

    PRINT_RESULTS();
}
