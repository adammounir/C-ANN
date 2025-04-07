// neural_network.c
#include "neural_network.h"
#include "mnist.h"

// Definition of global variables

double expectedLayer[numOutput];
double hiddenLayer1[numHidden1];
double hiddenLayer2[numHidden2];
double inputLayer[numInput];
double outputLayer[numOutput];

double inputToHidden1[numHidden1][numInput];
double hidden1ToHidden2[numHidden2][numHidden1];
double hidden2ToOutput[numOutput][numHidden2];

double hidden1Bias[numHidden1];
double hidden2Bias[numHidden2];
double outputBias[numOutput];

double gradWeightsHidden2ToOutput[numOutput][numHidden2];
double gradWeightsInputToHidden1[numHidden1][numInput];
double gradWeightsHidden1ToHidden2[numHidden2][numHidden1];

double gradBiasesOuputLayer[numOutput];
double gradBiasesHiddenLayer1[numHidden1];
double gradBiasesHiddenLayer2[numHidden2];

// Applies the softmax function to the output layer
void softmax(double input[], int size)
{
    double sum = 0.0;
    double max = input[0];

    for (int i = 1; i < size; i++)
        if (input[i] > max) max = input[i];

    for (int i = 0; i < size; i++)
        sum += exp(input[i] - max);

    for (int i = 0; i < size; i++)
        input[i] = exp(input[i] - max) / sum;
}

// Returns the index of the maximum value in an array
int maxValueInArray_Index(double array[], int sizearray)
{
    int max = 0;
    for (int i = 1; i < sizearray; i++)
    {
        if (array[i] > array[max])
            max = i;
    }
    return max;
}

// Evaluates the accuracy of the model
void evaluateAccuracy()
{
    int correct = 0;

    for (int i = 0; i < NUM_TEST; i++)
    {
        for (int pix = 0; pix < SIZE; pix++)
            inputLayer[pix] = ((double)(test_image_char[i][pix])) / 255.0;

        frontPropagation();
        softmax(outputLayer, numOutput);

        int prediction = maxValueInArray_Index(outputLayer, numOutput);
        if (prediction == test_label[i])
            correct++;
    }

    double accuracy = (double)correct / NUM_TEST * 100.0;
    printf("\n[ACCURACY] Test set accuracy: \033[1;32m%.2f%%\033[0m (%d/%d)\n", accuracy, correct, NUM_TEST);
}

