#ifndef CONFIG_H
#define CONFIG_H

// Dataset configuration
#define MNIST_TRAIN_SIZE 60000
#define MNIST_TEST_SIZE 10000
#define MNIST_INPUT_SIZE 784
#define MNIST_PIXEL_WIDTH 28
#define MNIST_NUM_CLASSES 10
#define MNIST_LABEL_DIMS 1

// Model architecture
#define MODEL_HIDDEN_SIZE 16

// Training configuration
#define DEFAULT_EPOCHS 10
#define DEFAULT_BATCH_SIZE 50
#define DEFAULT_LEARNING_RATE 0.01f

#endif // CONFIG_H
