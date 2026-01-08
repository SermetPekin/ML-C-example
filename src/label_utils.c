#include <stdio.h>
#include <math.h>
#include "label_utils.h"

label_format_type label_detect_format(
    const matrix* labels,
    u32 num_classes
) {
    if (labels == NULL || labels->rows == 0) {
        return LABEL_FORMAT_AUTO;
    }

    // Single column (likely integer indices)
    if (labels->cols == 1) {
        // Verify integer values in valid range
        b32 all_integers = 1;
        b32 all_valid_range = 1;

        for (u32 i = 0; i < labels->rows; i++) {
            f32 val = labels->data[i];

            // Check if integer (FP precision)
            if (fabsf(val - roundf(val)) > 1e-6f) {
                all_integers = 0;
            }

            // Verify valid class range
            u32 idx = (u32)roundf(val);
            if (idx >= num_classes) {
                all_valid_range = 0;
            }
        }

        if (all_integers && all_valid_range) {
            return LABEL_FORMAT_INTEGER_INDICES;
        }
    }

    // Check for one-hot encoding
    if (labels->cols == num_classes) {
        b32 all_binary = 1;

        for (u32 i = 0; i < labels->rows * labels->cols; i++) {
            f32 val = labels->data[i];
            if (fabsf(val - 0.0f) > 1e-6f && fabsf(val - 1.0f) > 1e-6f) {
                all_binary = 0;
                break;
            }
        }

        if (all_binary) {
            return LABEL_FORMAT_ONE_HOT;
        }
    }

    // Default to raw floats
    return LABEL_FORMAT_RAW_FLOATS;
}

matrix* label_convert_to_onehot(
    mem_arena* arena,
    const matrix* labels,
    label_format_type format,
    u32 num_classes
) {
    if (labels == NULL) {
        fprintf(stderr, "Error: labels matrix is NULL\n");
        return NULL;
    }

    // Return if already one-hot
    if (format == LABEL_FORMAT_ONE_HOT ||
        (format == LABEL_FORMAT_AUTO && labels->cols == num_classes)) {
        return (matrix*)labels;  // Cast away const since caller needs mutable
    }

    // Convert integer indices to one-hot
    if (format == LABEL_FORMAT_INTEGER_INDICES ||
        (format == LABEL_FORMAT_AUTO && labels->cols == 1)) {

        matrix* onehot = mat_create(arena, labels->rows, num_classes);
        if (onehot == NULL) {
            fprintf(stderr, "Error: Failed to allocate one-hot matrix\n");
            return NULL;
        }

        // Zero initialize
        for (u32 i = 0; i < onehot->rows * onehot->cols; i++) {
            onehot->data[i] = 0.0f;
        }

        // Set one-hot entries
        for (u32 i = 0; i < labels->rows; i++) {
            u32 class_idx = (u32)roundf(labels->data[i]);

            if (class_idx >= num_classes) {
                fprintf(stderr,
                    "Error: Invalid label value %u at index %u (expected < %u)\n",
                    class_idx, i, num_classes);
                return NULL;
            }

            onehot->data[i * num_classes + class_idx] = 1.0f;
        }

        return onehot;
    }

    // Return as-is for raw floats (assumes correct dims)
    if (labels->cols == num_classes) {
        return (matrix*)labels;
    }

    fprintf(stderr,
        "Error: Cannot convert labels with %u columns to %u classes\n",
        labels->cols, num_classes);
    return NULL;
}

b32 label_validate(
    const matrix* labels,
    label_format_type format,
    u32 num_classes
) {
    if (labels == NULL || labels->rows == 0) {
        fprintf(stderr, "Error: labels matrix is NULL or empty\n");
        return 0;
    }

    if (format == LABEL_FORMAT_INTEGER_INDICES) {
        // Integer indices: single column [0, num_classes)
        if (labels->cols != 1) {
            fprintf(stderr,
                "Error: integer_indices format expects 1 column, got %u\n",
                labels->cols);
            return 0;
        }

        for (u32 i = 0; i < labels->rows; i++) {
            f32 val = labels->data[i];
            u32 idx = (u32)roundf(val);

            if (idx >= num_classes) {
                fprintf(stderr,
                    "Error: Label value %u at index %u exceeds num_classes=%u\n",
                    idx, i, num_classes);
                return 0;
            }
        }

        return 1;
    }

    if (format == LABEL_FORMAT_ONE_HOT) {
        // One-hot: num_classes columns with 0/1
        if (labels->cols != num_classes) {
            fprintf(stderr,
                "Error: one_hot format expects %u columns, got %u\n",
                num_classes, labels->cols);
            return 0;
        }

        // Each row must have exactly one 1
        for (u32 i = 0; i < labels->rows; i++) {
            u32 ones_count = 0;
            for (u32 j = 0; j < labels->cols; j++) {
                f32 val = labels->data[i * labels->cols + j];
                if (fabsf(val - 1.0f) < 1e-6f) {
                    ones_count++;
                } else if (fabsf(val - 0.0f) > 1e-6f) {
                    fprintf(stderr,
                        "Error: one_hot format expects 0 or 1, got %f at row %u col %u\n",
                        val, i, j);
                    return 0;
                }
            }

            if (ones_count != 1) {
                fprintf(stderr,
                    "Error: one_hot row %u has %u ones (expected 1)\n",
                    i, ones_count);
                return 0;
            }
        }

        return 1;
    }

    if (format == LABEL_FORMAT_RAW_FLOATS) {
        // Verify dimension match
        if (labels->cols != num_classes) {
            fprintf(stderr,
                "Error: raw_floats format expects %u columns, got %u\n",
                num_classes, labels->cols);
            return 0;
        }

        return 1;
    }

    fprintf(stderr, "Error: Unknown label format %d\n", format);
    return 0;
}
