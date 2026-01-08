#ifndef LABEL_UTILS_H
#define LABEL_UTILS_H

#include "matrix.h"
#include "utils/arena.h"

typedef enum {
    LABEL_FORMAT_INTEGER_INDICES,  // Single column, integer class indices
    LABEL_FORMAT_ONE_HOT,          // num_classes columns, 0/1 values
    LABEL_FORMAT_RAW_FLOATS,       // Any dimensionality, float values
    LABEL_FORMAT_AUTO              // Auto-detect from data
} label_format_type;

// Auto-detect label format from matrix contents
// Returns the detected format, or LABEL_FORMAT_AUTO if ambiguous
label_format_type label_detect_format(
    const matrix* labels,
    u32 num_classes
);

// Convert labels to one-hot encoding (required for training)
// If already one-hot, returns pointer to labels matrix (no copy)
// If integer indices, allocates new matrix with one-hot encoding
// Returns NULL on error
matrix* label_convert_to_onehot(
    mem_arena* arena,
    const matrix* labels,
    label_format_type format,
    u32 num_classes
);

// Validate labels are consistent with format and num_classes
// Returns true if valid, false if validation fails
b32 label_validate(
    const matrix* labels,
    label_format_type format,
    u32 num_classes
);

#endif // LABEL_UTILS_H
