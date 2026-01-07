#ifndef MODEL_H
#define MODEL_H

#include "utils/base.h"
#include "utils/arena.h"
#include "matrix.h"

typedef enum {
    MV_FLAG_NONE = 0,

    MV_FLAG_REQUIRES_GRAD  = (1 << 0),
    MV_FLAG_PARAMETER      = (1 << 1),
    MV_FLAG_INPUT          = (1 << 2),
    MV_FLAG_OUTPUT         = (1 << 3),
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),
    MV_FLAG_COST           = (1 << 5),
} model_var_flags;

typedef enum {
    MV_OP_NULL = 0,
    MV_OP_CREATE,

    _MV_OP_UNARY_START,

    MV_OP_RELU,
    MV_OP_SOFTMAX,

    _MV_OP_BINARY_START,

    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
} model_var_op;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

typedef struct model_var {
    u32 index;
    u32 flags;

    matrix* val;
    matrix* grad;

    model_var_op op;
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

// Flag checking helpers
static inline b32 mv_requires_grad(const model_var* mv) {
    return (mv->flags & MV_FLAG_REQUIRES_GRAD) == MV_FLAG_REQUIRES_GRAD;
}

static inline b32 mv_is_parameter(const model_var* mv) {
    return (mv->flags & MV_FLAG_PARAMETER) == MV_FLAG_PARAMETER;
}

static inline b32 mv_is_input(const model_var* mv) {
    return (mv->flags & MV_FLAG_INPUT) == MV_FLAG_INPUT;
}

static inline b32 mv_is_output(const model_var* mv) {
    return (mv->flags & MV_FLAG_OUTPUT) == MV_FLAG_OUTPUT;
}

static inline b32 mv_is_desired_output(const model_var* mv) {
    return (mv->flags & MV_FLAG_DESIRED_OUTPUT) == MV_FLAG_DESIRED_OUTPUT;
}

static inline b32 mv_is_cost(const model_var* mv) {
    return (mv->flags & MV_FLAG_COST) == MV_FLAG_COST;
}

typedef struct {
    model_var** vars;
    u32 size;
} model_program;

typedef struct {
    u32 num_vars;

    model_var* input;
    model_var* output;
    model_var* desired_output;
    model_var* cost;

    model_program forward_prog;
    model_program cost_prog;
} model_context;

typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;

// Variable operations
model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
);

model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);

model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);

model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

model_var* mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
);

// Program operations
model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
);

b32 model_prog_compute(model_program* prog);
void model_prog_compute_grads(model_program* prog);

// Model lifecycle
model_context* model_create(mem_arena* arena);
void model_compile(mem_arena* arena, model_context* model);
b32 model_feedforward(model_context* model);
void model_train(
    model_context* model,
    const model_training_desc* training_desc
);

#endif // MODEL_H
