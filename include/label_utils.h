#ifndef LABEL_UTILS_H
#define LABEL_UTILS_H

#include "matrix.h"
#include "utils/arena.h"

typedef enum {
    LABEL_FORMAT_INTEGER_INDICES,
    LABEL_FORMAT_ONE_HOT,
    LABEL_FORMAT_RAW_FLOATS,
    LABEL_FORMAT_AUTO
} label_format_type;

label_format_type label_detect_format(
    const matrix* labels,
    u32 num_classes
);

// Convert to one-hot (or return input if already one-hot)
matrix* label_convert_to_onehot(
    mem_arena* arena,
    const matrix* labels,
    label_format_type format,
    u32 num_classes
);

b32 label_validate(
    const matrix* labels,
    label_format_type format,
    u32 num_classes
);

#endif // LABEL_UTILS_H
