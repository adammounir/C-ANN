// mnist.c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include "mnist.h"

unsigned char train_image_char[NUM_TRAIN][SIZE];
unsigned char test_image_char[NUM_TEST][SIZE];
unsigned char train_label_char[NUM_TRAIN][1];
unsigned char test_label_char[NUM_TEST][1];

double train_image[NUM_TRAIN][SIZE];
double test_image[NUM_TEST][SIZE];
int train_label[NUM_TRAIN];
int test_label[NUM_TEST];

unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE];
int width[MAX_NUM_OF_IMAGES], height[MAX_NUM_OF_IMAGES];
int info_image[LEN_INFO_IMAGE];
int info_label[LEN_INFO_LABEL];

void printImage(int indeximage)
{
    for (int i = 0; i < SIZE; i++) {
        if (i % 28 == 0)
            printf("\n");

        double pixel = (double)test_image_char[indeximage][i] / 255;

        if (pixel > 0.7)
            printf("\033[0;31m");
        else if (pixel > 0.3)
            printf("\033[1;31m");
        else
            printf("\033[0;37m");

        printf("\u25A0 ");
    }
    printf("\033[0;37m\n\n");
}

void FlipLong(unsigned char *ptr)
{
    unsigned char val;

    val = *ptr;
    *ptr = *(ptr + 3);
    *(ptr + 3) = val;

    ptr++;
    val = *ptr;
    *ptr = *(ptr + 1);
    *(ptr + 1) = val;
}

void read_mnist_char(char *file_path, int num_data, int len_info, int arr_n,
                     unsigned char data_char[][arr_n], int info_arr[])
{
    int i, fd;
    unsigned char *ptr;

    if ((fd = open(file_path, O_RDONLY)) == -1) {
        fprintf(stderr, "couldn't open image file\n");
        exit(-1);
    }

    read(fd, info_arr, len_info * sizeof(int));

    for (i = 0; i < len_info; i++) {
        ptr = (unsigned char *)(info_arr + i);
        FlipLong(ptr);
    }

    for (i = 0; i < num_data; i++)
        read(fd, data_char[i], arr_n * sizeof(unsigned char));

    close(fd);
}

void image_char2double(int num_data, unsigned char data_image_char[][SIZE], double data_image[][SIZE])
{
    for (int i = 0; i < num_data; i++)
        for (int j = 0; j < SIZE; j++)
            data_image[i][j] = (double)data_image_char[i][j] / 255.0;
}

void label_char2int(int num_data, unsigned char data_label_char[][1], int data_label[])
{
    for (int i = 0; i < num_data; i++)
        data_label[i] = (int)data_label_char[i][0];
}

void load_mnist()
{
    read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, LEN_INFO_IMAGE, SIZE, train_image_char, info_image);
    image_char2double(NUM_TRAIN, train_image_char, train_image);

    read_mnist_char(TEST_IMAGE, NUM_TEST, LEN_INFO_IMAGE, SIZE, test_image_char, info_image);
    image_char2double(NUM_TEST, test_image_char, test_image);

    read_mnist_char(TRAIN_LABEL, NUM_TRAIN, LEN_INFO_LABEL, 1, train_label_char, info_label);
    label_char2int(NUM_TRAIN, train_label_char, train_label);

    read_mnist_char(TEST_LABEL, NUM_TEST, LEN_INFO_LABEL, 1, test_label_char, info_label);
    label_char2int(NUM_TEST, test_label_char, test_label);
}

