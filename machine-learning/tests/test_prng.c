#include <stdio.h>
#include <string.h>
#include "utils/base.h"
#include "utils/prng.h"
#include "test_utils.h"

int main(void) {
    TEST_SUITE("PRNG (Pseudo-Random Number Generator) Tests");

    // Test 1: PRNG produces non-zero values
    {
        u32 val = prng_rand();
        ASSERT_TRUE(val != 0 || prng_rand() != 0); // At least one call should be non-zero
        TEST("PRNG produces values");
    }

    // Test 2: PRNG produces different values
    {
        u32 val1 = prng_rand();
        u32 val2 = prng_rand();
        ASSERT_TRUE(val1 != val2);
        TEST("PRNG produces different sequential values");
    }

    // Test 3: PRNG float is in range [0, 1)
    {
        f32 val = prng_randf();
        ASSERT_TRUE(val >= 0.0f && val < 1.0f);
        TEST("PRNG float value is in [0, 1) range");
    }

    // Test 4: Multiple PRNG float values are in range
    {
        int all_in_range = 1;
        for (int i = 0; i < 100; i++) {
            f32 val = prng_randf();
            if (!(val >= 0.0f && val < 1.0f)) {
                all_in_range = 0;
            }
        }
        ASSERT_TRUE(all_in_range);
        TEST("100 PRNG float values all in [0, 1) range");
    }

    // Test 5: PRNG produces variety of values
    {
        u32 vals[10];
        for (int i = 0; i < 10; i++) {
            vals[i] = prng_rand();
        }

        // Check that not all values are the same
        int all_same = 1;
        for (int i = 1; i < 10; i++) {
            if (vals[i] != vals[0]) {
                all_same = 0;
            }
        }
        ASSERT_TRUE(!all_same);
        TEST("PRNG produces variety of values");
    }

    printf("\n");
    PRINT_RESULTS();
}
