#include <stdio.h>
#include <string.h>

#include "model.h"
#include "matrix.h"
#include "utils/arena.h"
#include "utils/prng.h"

model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
) {
    model_var* out = PUSH_STRUCT(arena, model_var);

    out->index = model->num_vars++;
    out->flags = flags;
    out->op = MV_OP_CREATE;
    out->val = mat_create(arena, rows, cols);

    if (flags & MV_FLAG_REQUIRES_GRAD) {
        out->grad = mat_create(arena, rows, cols);
    }

    if (flags & MV_FLAG_INPUT) { model->input = out; }
    if (flags & MV_FLAG_OUTPUT) { model->output = out; }
    if (flags & MV_FLAG_DESIRED_OUTPUT) { model->desired_output = out; }
    if (flags & MV_FLAG_COST) { model->cost = out; }

    return out;
}

static model_var* _mv_unary_impl(
    mem_arena* arena, model_context* model,
    model_var* input, u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    if (mv_requires_grad(input)) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(arena, model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = input;

    return out;
}

static model_var* _mv_binary_impl(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b,
    u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    if (mv_requires_grad(a) || mv_requires_grad(b)) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* out = mv_create(arena, model, rows, cols, flags);

    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    return _mv_unary_impl(
        arena, model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_RELU
    );
}

model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    return _mv_unary_impl(
        arena, model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_SOFTMAX
    );
}

model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_ADD
    );
}

model_var* mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_SUB
    );
}

model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->cols != b->val->rows) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, b->val->cols,
        flags, MV_OP_MATMUL
    );
}

model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
) {
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
        return NULL;
    }

    return _mv_binary_impl(
        arena, model, p, q,
        p->val->rows, p->val->cols,
        flags, MV_OP_CROSS_ENTROPY
    );
}

model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
) {
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);

    b8* visited = PUSH_ARRAY(scratch.arena, b8, model->num_vars);

    u32 stack_size = 0;
    u32 out_size = 0;
    model_var** stack = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);
    model_var** out = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);

    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        model_var* cur = stack[--stack_size];

        if (cur->index >= model->num_vars) { continue; }

        if (visited[cur->index]) {
            if (out_size < model->num_vars) {
                out[out_size++] = cur;
            }
            continue;
        }

        visited[cur->index] = true;

        if (stack_size < model->num_vars) {
            stack[stack_size++] = cur;
        }

        u32 num_inputs = MV_NUM_INPUTS(cur->op);
        for (u32 i = 0; i < num_inputs; i++) {
            model_var* input = cur->inputs[i];

            if (input->index >= model->num_vars || visited[input->index]) {
                continue;
            }

            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == input) {
                    for (u32 k = j; k < stack_size-1; k++) {
                        stack[k] = stack[k+1];
                    }
                    stack_size--;
                }
            }

            if (stack_size < model->num_vars) {
                stack[stack_size++] = input;
            }
        }
    }

    model_program prog = {
        .size = out_size,
        .vars = PUSH_ARRAY_NZ(arena, model_var*, out_size)
    };

    memcpy(prog.vars, out, sizeof(model_var*) * out_size);

    arena_scratch_release(scratch);

    return prog;
}

b32 model_prog_compute(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* cur = prog->vars[i];

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE: break;

            case _MV_OP_UNARY_START: break;

            case MV_OP_RELU: {
                if (!mat_relu(cur->val, a->val)) {
                    fprintf(stderr, "Error: ReLU operation failed at variable index %u\n", cur->index);
                    return false;
                }
            } break;

            case MV_OP_SOFTMAX: {
                if (!mat_softmax(cur->val, a->val)) {
                    fprintf(stderr, "Error: Softmax operation failed at variable index %u\n", cur->index);
                    return false;
                }
            } break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: {
                if (!mat_add(cur->val, a->val, b->val)) {
                    fprintf(stderr, "Error: Matrix addition failed at variable index %u\n", cur->index);
                    return false;
                }
            } break;

            case MV_OP_SUB: {
                if (!mat_sub(cur->val, a->val, b->val)) {
                    fprintf(stderr, "Error: Matrix subtraction failed at variable index %u\n", cur->index);
                    return false;
                }
            } break;

            case MV_OP_MATMUL: {
                if (!mat_mul(cur->val, a->val, b->val, 1, 0, 0)) {
                    fprintf(stderr, "Error: Matrix multiplication failed at variable index %u\n", cur->index);
                    return false;
                }
            } break;

            case MV_OP_CROSS_ENTROPY: {
                if (!mat_cross_entropy(cur->val, a->val, b->val)) {
                    fprintf(stderr, "Error: Cross entropy operation failed at variable index %u\n", cur->index);
                    return false;
                }
            } break;
        }
    }
    return true;
}

void model_prog_compute_grads(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* cur = prog->vars[i];

        if (!mv_requires_grad(cur)) {
            continue;
        }

        if (mv_is_parameter(cur)) {
            continue;
        }

        mat_clear(cur->grad);
    }

    mat_fill(prog->vars[prog->size-1]->grad, 1.0f);

    for (i64 i = (i64)prog->size - 1; i >= 0; i--) {
        model_var* cur = prog->vars[i];

        if (!mv_requires_grad(cur)) {
            continue;
        }

        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];

        u32 num_inputs = MV_NUM_INPUTS(cur->op);

        if (num_inputs == 1 && !mv_requires_grad(a)) {
            continue;
        }

        if (num_inputs == 2 && !mv_requires_grad(a) && !mv_requires_grad(b)) {
            continue;
        }

        switch (cur->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE: break;

            case _MV_OP_UNARY_START: break;

            case MV_OP_RELU: {
                mat_relu_add_grad(a->grad, a->val, cur->grad);
            } break;

            case MV_OP_SOFTMAX: {
                mat_softmax_add_grad(a->grad, cur->val, cur->grad);
            } break;

            case _MV_OP_BINARY_START: break;

            case MV_OP_ADD: {
                if (mv_requires_grad(a)) {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                if (mv_requires_grad(b)) {
                    mat_add(b->grad, b->grad, cur->grad);
                }
            } break;

            case MV_OP_SUB: {
                if (mv_requires_grad(a)) {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                if (mv_requires_grad(b)) {
                    mat_sub(b->grad, b->grad, cur->grad);
                }
            } break;

            case MV_OP_MATMUL: {
                if (mv_requires_grad(a)) {
                    mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
                }

                if (mv_requires_grad(b)) {
                    mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
                }
            } break;

            case MV_OP_CROSS_ENTROPY: {
                model_var* p = a;
                model_var* q = b;

                mat_cross_entropy_add_grad(
                    p->grad, q->grad, p->val, q->val, cur->grad
                );
            } break;
        }
    }
}

model_context* model_create(mem_arena* arena) {
    model_context* model = PUSH_STRUCT(arena, model_context);

    return model;
}

void model_compile(mem_arena* arena, model_context* model) {
    if (model->output != NULL) {
        model->forward_prog = model_prog_create(arena, model, model->output);
    }

    if (model->cost != NULL) {
        model->cost_prog = model_prog_create(arena, model, model->cost);
    }
}

b32 model_feedforward(model_context* model) {
    if (!model_prog_compute(&model->forward_prog)) {
        fprintf(stderr, "Error: Forward pass computation failed.\n");
        return false;
    }
    return true;
}

// Initialize Adam optimizer state for all parameters
static void _adam_init(mem_arena* arena, model_context* model) {
    for (u32 i = 0; i < model->cost_prog.size; i++) {
        model_var* cur = model->cost_prog.vars[i];

        if (!mv_is_parameter(cur)) {
            continue;
        }

        cur->adam_m = mat_create(arena, cur->val->rows, cur->val->cols);
        cur->adam_v = mat_create(arena, cur->val->rows, cur->val->cols);
    }
}

// Perform SGD parameter update
static void _update_params_sgd(
    model_context* model,
    const model_training_desc* training_desc
) {
    for (u32 i = 0; i < model->cost_prog.size; i++) {
        model_var* cur = model->cost_prog.vars[i];

        if (!mv_is_parameter(cur)) {
            continue;
        }

        mat_scale(
            cur->grad,
            training_desc->learning_rate /
            training_desc->batch_size
        );
        mat_sub(cur->val, cur->val, cur->grad);
    }
}

// Perform Adam parameter update
static void _update_params_adam(
    model_context* model,
    const model_training_desc* training_desc,
    u32 step
) {
    f32 beta1 = training_desc->adam_beta1;
    f32 beta2 = training_desc->adam_beta2;
    f32 epsilon = training_desc->adam_epsilon;
    f32 lr = training_desc->learning_rate;
    f32 batch_size = (f32)training_desc->batch_size;

    // Bias correction terms
    f32 bias_correction1 = 1.0f - powf(beta1, (f32)step);
    f32 bias_correction2 = 1.0f - powf(beta2, (f32)step);
    f32 bias_correction = sqrtf(bias_correction2) / bias_correction1;

    for (u32 i = 0; i < model->cost_prog.size; i++) {
        model_var* cur = model->cost_prog.vars[i];

        if (!mv_is_parameter(cur)) {
            continue;
        }

        // Scale gradient by batch size
        mat_scale(cur->grad, 1.0f / batch_size);

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
        for (u32 j = 0; j < cur->grad->rows * cur->grad->cols; j++) {
            cur->adam_m->data[j] =
                beta1 * cur->adam_m->data[j] +
                (1.0f - beta1) * cur->grad->data[j];
        }

        // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
        for (u32 j = 0; j < cur->grad->rows * cur->grad->cols; j++) {
            f32 g = cur->grad->data[j];
            cur->adam_v->data[j] =
                beta2 * cur->adam_v->data[j] +
                (1.0f - beta2) * g * g;
        }

        // Update parameters: param = param - lr * bias_correction * m / (sqrt(v) + eps)
        for (u32 j = 0; j < cur->val->rows * cur->val->cols; j++) {
            f32 m_hat = cur->adam_m->data[j];
            f32 v_hat = cur->adam_v->data[j];
            f32 update = lr * bias_correction * m_hat / (sqrtf(v_hat) + epsilon);
            cur->val->data[j] -= update;
        }
    }
}

void model_train(
    model_context* model,
    const model_training_desc* training_desc
) {
    matrix* train_images = training_desc->train_images;
    matrix* train_labels = training_desc->train_labels;
    matrix* test_images = training_desc->test_images;
    matrix* test_labels = training_desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;
    u32 num_tests = test_images->rows;

    u32 num_batches = num_examples / training_desc->batch_size;

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    // Initialize Adam state if using Adam optimizer
    if (training_desc->optimizer == OPTIMIZER_ADAM) {
        _adam_init(scratch.arena, model);
    }

    u32* training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);
    for (u32 i = 0; i < num_examples; i++) {
        training_order[i] = i;
    }

    u32 step_counter = 1;  // For Adam bias correction (1-indexed)

    for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
        for (u32 i = 0; i < num_examples; i++) {
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;

            u32 tmp = training_order[b];
            training_order[b] = training_order[a];
            training_order[a] = tmp;
        }

        for (u32 batch = 0; batch < num_batches; batch++) {
            for (u32 i = 0; i < model->cost_prog.size; i++) {
                model_var* cur = model->cost_prog.vars[i];

                if (mv_is_parameter(cur)) {
                    mat_clear(cur->grad);
                }
            }

            f32 avg_cost = 0.0f;
            for (u32 i = 0; i < training_desc->batch_size; i++) {
                u32 order_index = batch * training_desc->batch_size + i;
                u32 index = training_order[order_index];

                memcpy(
                    model->input->val->data,
                    train_images->data + index * input_size,
                    sizeof(f32) * input_size
                );

                memcpy(
                    model->desired_output->val->data,
                    train_labels->data + index * output_size,
                    sizeof(f32) * output_size
                );

                if (!model_prog_compute(&model->cost_prog)) {
                    fprintf(stderr, "Error: Cost computation failed during training at epoch %u, batch %u, sample %u\n",
                            epoch, batch, i);
                    arena_scratch_release(scratch);
                    return;
                }
                model_prog_compute_grads(&model->cost_prog);

                avg_cost += mat_sum(model->cost->val);
            }
            avg_cost /= (f32)training_desc->batch_size;

            // Update parameters using the selected optimizer
            if (training_desc->optimizer == OPTIMIZER_SGD) {
                _update_params_sgd(model, training_desc);
            } else if (training_desc->optimizer == OPTIMIZER_ADAM) {
                _update_params_adam(model, training_desc, step_counter);
                step_counter++;
            }

            printf(
                "Epoch %2d / %2d, Batch %4d / %4d, Average Cost: %.4f\r",
                epoch + 1, training_desc->epochs,
                batch + 1, num_batches, avg_cost
            );
            fflush(stdout);
        }
        printf("\n");

        u32 num_correct = 0;
        f32 avg_cost = 0;
        for (u32 i = 0; i < num_tests; i++) {
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
                fprintf(stderr, "Error: Cost computation failed during testing at epoch %u, test sample %u\n",
                        epoch, i);
                arena_scratch_release(scratch);
                return;
            }

            avg_cost += mat_sum(model->cost->val);
            num_correct +=
                mat_argmax(model->output->val) ==
                mat_argmax(model->desired_output->val);
        }

        avg_cost /= (f32)num_tests;
        printf(
            "Test Completed. Accuracy: %5d / %5d (%.1f%%), Average Cost: %.4f\n",
            num_correct, num_tests, (f32)num_correct / num_tests * 100.0f,
            avg_cost
        );
    }

    arena_scratch_release(scratch);
}
