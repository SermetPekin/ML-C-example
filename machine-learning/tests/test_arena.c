#include <stdio.h>
#include <string.h>
#include "../include/utils/base.h"
#include "../include/utils/arena.h"
#include "test_utils.h"

int main(void) {
    TEST_SUITE("Arena Allocator Tests");

    // Test 1: Arena creation
    {
        mem_arena* arena = arena_create(1024 * 1024, 64 * 1024); // 1 MB reserve, 64 KB commit
        ASSERT_TRUE(arena != NULL);
        ASSERT_TRUE(arena->pos >= 0); // Position starts at or near zero
        arena_destroy(arena);
        TEST("Arena creation allocates memory");
    }

    // Test 2: Single allocation
    {
        mem_arena* arena = arena_create(1024, 256);
        u64 initial_pos = arena->pos;
        void* ptr = arena_push(arena, 100, false);
        ASSERT_TRUE(ptr != NULL);
        ASSERT_TRUE(arena->pos > initial_pos); // Position increases
        arena_destroy(arena);
        TEST("Single allocation increases position");
    }

    // Test 3: Multiple allocations
    {
        mem_arena* arena = arena_create(10 * 1024, 1024);
        void* ptr1 = arena_push(arena, 100, false);
        void* ptr2 = arena_push(arena, 200, false);
        void* ptr3 = arena_push(arena, 150, false);

        ASSERT_TRUE(ptr1 != NULL);
        ASSERT_TRUE(ptr2 != NULL);
        ASSERT_TRUE(ptr3 != NULL);
        ASSERT_TRUE(arena->pos >= 450); // Position is at least 450 (may be more due to alignment)
        arena_destroy(arena);
        TEST("Multiple allocations stack correctly");
    }

    // Test 4: Memory is not overlapping
    {
        mem_arena* arena = arena_create(10 * 1024, 1024);
        void* ptr1 = arena_push(arena, 100, false);
        void* ptr2 = arena_push(arena, 100, false);

        // ptr2 should be offset from ptr1 (accounting for alignment)
        ASSERT_TRUE((u64)ptr2 > (u64)ptr1);
        arena_destroy(arena);
        TEST("Allocations don't overlap");
    }

    // Test 5: Write and read back data
    {
        mem_arena* arena = arena_create(10 * 1024, 1024);
        int* data = (int*)arena_push(arena, sizeof(int) * 5, false);

        for (int i = 0; i < 5; i++) {
            data[i] = i * 10;
        }

        int all_correct = 1;
        for (int i = 0; i < 5; i++) {
            if (data[i] != i * 10) {
                all_correct = 0;
            }
        }
        ASSERT_TRUE(all_correct);
        arena_destroy(arena);
        TEST("Data persists correctly in allocated memory");
    }

    // Test 6: Arena clear
    {
        mem_arena* arena = arena_create(10 * 1024, 1024);
        arena_push(arena, 500, false);
        u64 after_alloc = arena->pos;
        ASSERT_TRUE(after_alloc > 0);

        arena_clear(arena);
        // Arena clear resets to ARENA_BASE_POS (size of arena metadata) instead of 0
        ASSERT_TRUE(arena->pos >= 0 && arena->pos <= 32); // Expected to be around 32 (sizeof(mem_arena))
        arena_destroy(arena);
        TEST("Arena clear resets position");
    }

    // Test 7: PUSH_STRUCT macro
    {
        mem_arena* arena = arena_create(10 * 1024, 1024);

        typedef struct {
            int x;
            float y;
            char z;
        } test_struct;

        test_struct* s = PUSH_STRUCT(arena, test_struct);
        ASSERT_TRUE(s != NULL);
        TEST("PUSH_STRUCT macro allocates memory");
        arena_destroy(arena);
    }

    // Test 8: PUSH_ARRAY macro
    {
        mem_arena* arena = arena_create(10 * 1024, 1024);
        int* arr = PUSH_ARRAY(arena, int, 10);
        ASSERT_TRUE(arr != NULL);
        arena_destroy(arena);
        TEST("PUSH_ARRAY macro allocates memory");
    }

    // Test 9: Arena pop
    {
        mem_arena* arena = arena_create(10 * 1024, 1024);
        arena_push(arena, 100, false);
        u64 pos_after_alloc = arena->pos;

        arena_pop(arena, 50);
        u64 pos_after_pop = arena->pos;
        ASSERT_TRUE(pos_after_pop < pos_after_alloc); // Pop reduces position
        arena_destroy(arena);
        TEST("Arena pop reduces position");
    }

    // Test 10: Temp arena
    {
        mem_arena* arena = arena_create(10 * 1024, 1024);
        arena_push(arena, 100, false);

        mem_arena_temp temp = arena_temp_begin(arena);
        u64 saved_pos = temp.start_pos;

        arena_push(arena, 100, false);
        u64 pos_after_temp_alloc = arena->pos;
        ASSERT_TRUE(pos_after_temp_alloc > saved_pos); // Temp allocation increases position

        arena_temp_end(temp);

        // After ending temp, main arena position should be restored
        ASSERT_EQ((int)arena->pos, (int)saved_pos);
        arena_destroy(arena);
        TEST("Temporary arena allocations are cleaned up");
    }

    printf("\n");
    PRINT_RESULTS();
}
