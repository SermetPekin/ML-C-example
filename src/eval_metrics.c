#include <stdio.h>
#include <string.h>
#include <math.h>

#include "eval_metrics.h"
#include "matrix.h"
#include "utils/arena.h"

confusion_matrix* cm_create(mem_arena* arena, u32 num_classes) {
    confusion_matrix* cm = PUSH_STRUCT(arena, confusion_matrix);
    cm->num_classes = num_classes;
    cm->data = PUSH_ARRAY_NZ(arena, u32, num_classes * num_classes);
    return cm;
}

void cm_add(confusion_matrix* cm, u32 actual, u32 predicted) {
    if (actual < cm->num_classes && predicted < cm->num_classes) {
        cm->data[actual * cm->num_classes + predicted]++;
    }
}

static void _calculate_class_metrics(
    confusion_matrix* cm,
    class_metrics* metrics
) {
    for (u32 class = 0; class < cm->num_classes; class++) {
        u32 tp = cm->data[class * cm->num_classes + class];
        u32 fp = 0, fn = 0;

        // Calculate FP and FN
        for (u32 i = 0; i < cm->num_classes; i++) {
            if (i != class) {
                // FP: predicted as class but actually something else
                fp += cm->data[i * cm->num_classes + class];
                // FN: predicted as something else but actually class
                fn += cm->data[class * cm->num_classes + i];
            }
        }

        // Calculate metrics
        f32 precision = (tp + fp > 0) ? (f32)tp / (tp + fp) : 0.0f;
        f32 recall = (tp + fn > 0) ? (f32)tp / (tp + fn) : 0.0f;
        f32 f1 = (precision + recall > 0) ?
            2.0f * (precision * recall) / (precision + recall) : 0.0f;

        metrics[class].precision = precision;
        metrics[class].recall = recall;
        metrics[class].f1_score = f1;
        metrics[class].count = tp + fn;
    }
}

evaluation_result* model_evaluate(
    mem_arena* arena,
    model_context* model,
    matrix* test_images,
    matrix* test_labels,
    u32 num_classes
) {
    evaluation_result* result = PUSH_STRUCT(arena, evaluation_result);
    result->num_classes = num_classes;
    result->cm = cm_create(arena, num_classes);
    result->class_metrics_arr = PUSH_ARRAY(arena, class_metrics, num_classes);

    u32 input_size = test_images->cols;
    u32 output_size = test_labels->cols;
    u32 num_tests = test_images->rows;

    u32 correct = 0;
    f32 total_loss = 0.0f;

    for (u32 i = 0; i < num_tests; i++) {
        // Run forward pass
        memcpy(
            model->input->val->data,
            test_images->data + i * input_size,
            sizeof(f32) * input_size
        );

        memcpy(
            model->desired_output->val->data,
            test_labels->data + i * output_size,
            sizeof(f32) * output_size
        );

        if (!model_prog_compute(&model->cost_prog)) {
            fprintf(stderr, "Error: Cost computation failed during evaluation at sample %u\n", i);
            return NULL;
        }

        // Get predictions
        u32 predicted = mat_argmax(model->output->val);
        u32 actual = mat_argmax(model->desired_output->val);

        cm_add(result->cm, actual, predicted);

        if (predicted == actual) {
            correct++;
        }

        total_loss += mat_sum(model->cost->val);
    }

    result->overall_accuracy = (f32)correct / num_tests;
    result->average_loss = total_loss / num_tests;

    _calculate_class_metrics(result->cm, result->class_metrics_arr);

    return result;
}

void eval_print_results(const evaluation_result* results) {
    printf("\n");
    printf("=====================================\n");
    printf("             EVALUATION RESULTS      \n");
    printf("=====================================\n\n");

    printf("Overall Accuracy: %.2f%%\n", results->overall_accuracy * 100.0f);
    printf("Average Loss: %.6f\n\n", results->average_loss);

    printf("%-10s %-12s %-12s %-12s %-8s\n",
           "Class", "Precision", "Recall", "F1-Score", "Support");
    printf("%-10s %-12s %-12s %-12s %-8s\n",
           "-----", "---------", "------", "--------", "-------");

    for (u32 i = 0; i < results->num_classes; i++) {
        class_metrics* cm = &results->class_metrics_arr[i];
        printf("%-10u %-12.4f %-12.4f %-12.4f %-8u\n",
               i,
               cm->precision,
               cm->recall,
               cm->f1_score,
               cm->count);
    }

    printf("\n");
    printf("Confusion Matrix:\n");
    printf("(rows=actual, cols=predicted)\n\n");

    // Print header
    printf("%-10s", "Actual\\Pred");
    for (u32 i = 0; i < results->num_classes; i++) {
        printf("%8u", i);
    }
    printf("\n");
    printf("%-10s", "----------");
    for (u32 i = 0; i < results->num_classes; i++) {
        printf("%8s", "--------");
    }
    printf("\n");

    // Print matrix
    for (u32 actual = 0; actual < results->num_classes; actual++) {
        printf("%-10u", actual);
        for (u32 pred = 0; pred < results->num_classes; pred++) {
            printf("%8u",
                   results->cm->data[actual * results->num_classes + pred]);
        }
        printf("\n");
    }

    printf("\n=====================================\n\n");
}

b32 eval_save_results(const evaluation_result* results, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' for writing\n", filename);
        return false;
    }

    fprintf(f, "=====================================\n");
    fprintf(f, "             EVALUATION RESULTS      \n");
    fprintf(f, "=====================================\n\n");

    fprintf(f, "Overall Accuracy: %.2f%%\n", results->overall_accuracy * 100.0f);
    fprintf(f, "Average Loss: %.6f\n\n", results->average_loss);

    fprintf(f, "%-10s %-12s %-12s %-12s %-8s\n",
            "Class", "Precision", "Recall", "F1-Score", "Support");
    fprintf(f, "%-10s %-12s %-12s %-12s %-8s\n",
            "-----", "---------", "------", "--------", "-------");

    for (u32 i = 0; i < results->num_classes; i++) {
        class_metrics* cm = &results->class_metrics_arr[i];
        fprintf(f, "%-10u %-12.4f %-12.4f %-12.4f %-8u\n",
                i,
                cm->precision,
                cm->recall,
                cm->f1_score,
                cm->count);
    }

    fprintf(f, "\n");
    fprintf(f, "Confusion Matrix:\n");
    fprintf(f, "(rows=actual, cols=predicted)\n\n");

    // Print header
    fprintf(f, "%-10s", "Actual\\Pred");
    for (u32 i = 0; i < results->num_classes; i++) {
        fprintf(f, "%8u", i);
    }
    fprintf(f, "\n");
    fprintf(f, "%-10s", "----------");
    for (u32 i = 0; i < results->num_classes; i++) {
        fprintf(f, "%8s", "--------");
    }
    fprintf(f, "\n");

    // Print matrix
    for (u32 actual = 0; actual < results->num_classes; actual++) {
        fprintf(f, "%-10u", actual);
        for (u32 pred = 0; pred < results->num_classes; pred++) {
            fprintf(f, "%8u",
                    results->cm->data[actual * results->num_classes + pred]);
        }
        fprintf(f, "\n");
    }

    fprintf(f, "\n=====================================\n\n");

    fclose(f);
    return true;
}
