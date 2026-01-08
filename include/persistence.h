#ifndef PERSISTENCE_H
#define PERSISTENCE_H

#include "model.h"
#include "utils/base.h"

// Binary format: header | metadata array | parameter data

typedef struct {
    u32 magic;          // 0x4D4C4E4E
    u32 version;        // Format version
    u32 num_parameters;
    u32 reserved;
} model_file_header;

typedef struct {
    u32 rows;
    u32 cols;
} parameter_metadata;

b32 model_save(const model_context* model, const char* filename);

// Model structure must match layer shapes in file
b32 model_load(model_context* model, const char* filename);

#endif // PERSISTENCE_H
