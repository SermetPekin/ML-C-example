#include <stdio.h>
#include <string.h>
#include "utils/base.h"
#include "utils/arena.h"
#include "matrix.h"
#include "test_utils.h"

int main(void) {
    TEST_SUITE("Matrix Operations Tests");

    // Create arena for test allocations
    mem_arena* arena = arena_create(256 * 1024 * 1024, 8 * 1024 * 1024); // 256 MB reserve, 8 MB commit

    // Test 1: Matrix creation and basic properties
    {
        matrix* m = mat_create(arena, 3, 4);
        ASSERT_EQ(m->rows, 3);
        ASSERT_EQ(m->cols, 4);
        TEST("Matrix creation with correct dimensions");
    }

    // Test 2: Matrix fill
    {
        matrix* m = mat_create(arena, 2, 2);
        mat_fill(m, 5.0f);
        for (u32 i = 0; i < m->rows * m->cols; i++) {
            ASSERT_FLOAT_EQ(m->data[i], 5.0f, 1e-6f);
        }
        TEST("Matrix fill with constant value");
    }

    // Test 3: Matrix clear
    {
        matrix* m = mat_create(arena, 2, 2);
        mat_fill(m, 5.0f);
        mat_clear(m);
        for (u32 i = 0; i < m->rows * m->cols; i++) {
            ASSERT_FLOAT_EQ(m->data[i], 0.0f, 1e-6f);
        }
        TEST("Matrix clear sets all values to zero");
    }

    // Test 4: Matrix addition
    {
        matrix* a = mat_create(arena, 2, 2);
        matrix* b = mat_create(arena, 2, 2);
        matrix* c = mat_create(arena, 2, 2);

        mat_fill(a, 1.0f);
        mat_fill(b, 2.0f);
        mat_clear(c);

        mat_add(c, a, b);

        for (u32 i = 0; i < c->rows * c->cols; i++) {
            ASSERT_FLOAT_EQ(c->data[i], 3.0f, 1e-6f);
        }
        TEST("Matrix addition (1 + 2 = 3)");
    }

    // Test 5: Matrix subtraction
    {
        matrix* a = mat_create(arena, 2, 2);
        matrix* b = mat_create(arena, 2, 2);
        matrix* c = mat_create(arena, 2, 2);

        mat_fill(a, 5.0f);
        mat_fill(b, 2.0f);
        mat_clear(c);

        mat_sub(c, a, b);

        for (u32 i = 0; i < c->rows * c->cols; i++) {
            ASSERT_FLOAT_EQ(c->data[i], 3.0f, 1e-6f);
        }
        TEST("Matrix subtraction (5 - 2 = 3)");
    }

    // Test 6: Matrix scale
    {
        matrix* m = mat_create(arena, 2, 2);
        mat_fill(m, 2.0f);
        mat_scale(m, 3.0f);

        for (u32 i = 0; i < m->rows * m->cols; i++) {
            ASSERT_FLOAT_EQ(m->data[i], 6.0f, 1e-6f);
        }
        TEST("Matrix scaling (2 * 3 = 6)");
    }

    // Test 7: Matrix multiplication (simple 2x2 case)
    {
        matrix* a = mat_create(arena, 2, 2);
        matrix* b = mat_create(arena, 2, 2);
        matrix* c = mat_create(arena, 2, 2);

        // a = [[1, 2], [3, 4]]
        a->data[0] = 1.0f; a->data[1] = 2.0f;
        a->data[2] = 3.0f; a->data[3] = 4.0f;

        // b = [[5, 6], [7, 8]]
        b->data[0] = 5.0f; b->data[1] = 6.0f;
        b->data[2] = 7.0f; b->data[3] = 8.0f;

        mat_mul(c, a, b, true, false, false);

        // c should be [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        // = [[19, 22], [43, 50]]
        ASSERT_FLOAT_EQ(c->data[0], 19.0f, 1e-5f);
        ASSERT_FLOAT_EQ(c->data[1], 22.0f, 1e-5f);
        ASSERT_FLOAT_EQ(c->data[2], 43.0f, 1e-5f);
        ASSERT_FLOAT_EQ(c->data[3], 50.0f, 1e-5f);
        TEST("Matrix multiplication 2x2");
    }

    // Test 8: Matrix sum
    {
        matrix* m = mat_create(arena, 2, 2);
        m->data[0] = 1.0f;
        m->data[1] = 2.0f;
        m->data[2] = 3.0f;
        m->data[3] = 4.0f;

        f32 sum = mat_sum(m);
        ASSERT_FLOAT_EQ(sum, 10.0f, 1e-6f);
        TEST("Matrix sum (1+2+3+4=10)");
    }

    // Test 9: Matrix argmax
    {
        matrix* m = mat_create(arena, 2, 3);
        m->data[0] = 1.0f;
        m->data[1] = 5.0f;
        m->data[2] = 2.0f;
        m->data[3] = 3.0f;
        m->data[4] = 4.0f;
        m->data[5] = 6.0f;

        u64 max_idx = mat_argmax(m);
        ASSERT_EQ((int)max_idx, 5);
        TEST("Matrix argmax finds maximum element index");
    }

    // Test 10: ReLU activation
    {
        matrix* in = mat_create(arena, 1, 4);
        matrix* out = mat_create(arena, 1, 4);
        in->data[0] = -1.0f;
        in->data[1] = 2.0f;
        in->data[2] = -3.0f;
        in->data[3] = 4.0f;

        mat_relu(out, in);

        ASSERT_FLOAT_EQ(out->data[0], 0.0f, 1e-6f);
        ASSERT_FLOAT_EQ(out->data[1], 2.0f, 1e-6f);
        ASSERT_FLOAT_EQ(out->data[2], 0.0f, 1e-6f);
        ASSERT_FLOAT_EQ(out->data[3], 4.0f, 1e-6f);
        TEST("ReLU activation (negative→0, positive→unchanged)");
    }

    arena_destroy(arena);

    printf("\n");
    PRINT_RESULTS();
}
