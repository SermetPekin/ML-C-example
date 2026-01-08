#ifndef EVAL_METRICS_H
#define EVAL_METRICS_H

#include "utils/base.h"
#include "utils/arena.h"
#include "matrix.h"
#include "model.h"

typedef struct {
    u32* data;      // confusion_matrix[actual][predicted]
    u32 num_classes;
} confusion_matrix;

typedef struct {
    f32 precision;
    f32 recall;
    f32 f1_score;
    u32 count;
} class_metrics;

typedef struct {
    f32 overall_accuracy;
    f32 average_loss;
    confusion_matrix* cm;
    class_metrics* class_metrics_arr;
    u32 num_classes;
} evaluation_result;

confusion_matrix* cm_create(mem_arena* arena, u32 num_classes);
void cm_add(confusion_matrix* cm, u32 actual, u32 predicted);

evaluation_result* model_evaluate(
    mem_arena* arena,
    model_context* model,
    matrix* test_images,
    matrix* test_labels,
    u32 num_classes
);

void eval_print_results(const evaluation_result* results);
b32 eval_save_results(const evaluation_result* results, const char* filename);

#endif // EVAL_METRICS_H
