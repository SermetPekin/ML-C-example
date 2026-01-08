#ifndef PERSISTENCE_H
#define PERSISTENCE_H

#include "model.h"
#include "utils/base.h"

// Binary model file format
// [Header]
// [Parameter metadata array]
// [Parameter data]

typedef struct {
    u32 magic;           // 0x4D4C4E4E ("MLNN")
    u32 version;         // Format version (currently 1)
    u32 num_parameters;  // Number of parameter tensors (weights + biases)
    u32 reserved;        // Reserved for future use
} model_file_header;

// Metadata for each parameter tensor
typedef struct {
    u32 rows;
    u32 cols;
} parameter_metadata;

// Save trained model weights to binary file
// Returns true on success, false on failure
b32 model_save(const model_context* model, const char* filename);

// Load trained model weights from binary file
// Model structure must already be created with correct layer shapes
// Returns true on success, false on failure
b32 model_load(model_context* model, const char* filename);

#endif // PERSISTENCE_H
