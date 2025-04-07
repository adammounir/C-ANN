// mnist.h
#ifndef MNIST_H
#define MNIST_H

#define SIZE 784  // 28x28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2
#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

#define TRAIN_IMAGE "./images_data_python/train-images.idx3-ubyte"
#define TRAIN_LABEL "./images_data_python/train-labels.idx1-ubyte"
#define TEST_IMAGE  "./images_data_python/t10k-images.idx3-ubyte"
#define TEST_LABEL  "./images_data_python/t10k-labels.idx1-ubyte"

extern unsigned char train_image_char[NUM_TRAIN][SIZE];
extern unsigned char test_image_char[NUM_TEST][SIZE];
extern unsigned char train_label_char[NUM_TRAIN][1];
extern unsigned char test_label_char[NUM_TEST][1];

extern double train_image[NUM_TRAIN][SIZE];
extern double test_image[NUM_TEST][SIZE];
extern int train_label[NUM_TRAIN];
extern int test_label[NUM_TEST];

void load_mnist();
void printImage(int index);

#endif // MNIST_H

