#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int total;
    int passed;
    int failed;
} test_stats;

static test_stats stats = {0, 0, 0};

#define ASSERT_EQ(actual, expected) \
    do { \
        stats.total++; \
        if ((actual) == (expected)) { \
            stats.passed++; \
        } else { \
            stats.failed++; \
            printf("  ✗ FAILED: %s:%d\n", __FILE__, __LINE__); \
            printf("    Expected: %d, Got: %d\n", expected, actual); \
        } \
    } while(0)

#define ASSERT_FLOAT_EQ(actual, expected, epsilon) \
    do { \
        stats.total++; \
        float diff = fabsf((actual) - (expected)); \
        if (diff < (epsilon)) { \
            stats.passed++; \
        } else { \
            stats.failed++; \
            printf("  ✗ FAILED: %s:%d\n", __FILE__, __LINE__); \
            printf("    Expected: %f, Got: %f (diff: %f)\n", expected, actual, diff); \
        } \
    } while(0)

#define ASSERT_TRUE(condition) \
    do { \
        stats.total++; \
        if ((condition)) { \
            stats.passed++; \
        } else { \
            stats.failed++; \
            printf("  ✗ FAILED: %s:%d\n", __FILE__, __LINE__); \
            printf("    Assertion failed: %s\n", #condition); \
        } \
    } while(0)

#define TEST_SUITE(name) \
    printf("\n%s\n", name); \
    printf("======================\n")

#define TEST(name) \
    printf("  ✓ %s\n", name)

#define PRINT_RESULTS() \
    do { \
        printf("\n======================\n"); \
        printf("Test Results: %d/%d passed\n", stats.passed, stats.total); \
        if (stats.failed > 0) { \
            printf("FAILED: %d test(s)\n", stats.failed); \
            return 1; \
        } else { \
            printf("All tests passed!\n"); \
            return 0; \
        } \
    } while(0)

#endif
