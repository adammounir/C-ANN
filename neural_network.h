// neural_network.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Layer sizes
#define numInput 784
#define numHidden1 64
#define numHidden2 64
#define numOutput 10

// Declaration of forward propagation output vectors
extern double expectedLayer[numOutput];
extern double hiddenLayer1[numHidden1];
extern double hiddenLayer2[numHidden2];
extern double inputLayer[numInput];
extern double outputLayer[numOutput];

// Weights
extern double inputToHidden1[numHidden1][numInput];
extern double hidden1ToHidden2[numHidden2][numHidden1];
extern double hidden2ToOutput[numOutput][numHidden2];

// Biases
extern double hidden1Bias[numHidden1];
extern double hidden2Bias[numHidden2];
extern double outputBias[numOutput];

// Gradients
extern double gradWeightsHidden2ToOutput[numOutput][numHidden2];
extern double gradWeightsInputToHidden1[numHidden1][numInput];
extern double gradWeightsHidden1ToHidden2[numHidden2][numHidden1];

extern double gradBiasesOuputLayer[numOutput];
extern double gradBiasesHiddenLayer1[numHidden1];
extern double gradBiasesHiddenLayer2[numHidden2];

// Functions
void trainNetwork();
void softmax(double input[], int size);
int maxValueInArray_Index(double array[], int sizearray);
void evaluateAccuracy();

// Model Saving and Loading
void saveModel(const char *filename);
void loadModel(const char *filename);

// Forward propagation
void frontPropagation();
double sigmoid(double x);
double relu(double x);
double randomnumber();
void assignRandomToMatrix(size_t currentdim, size_t prevdim, double layer[currentdim][prevdim]);
void assignRandomToVector(double layer[], int layerSize);
void layerValues(double previousValues[], double currentValues[], double currentBias[], size_t sizePreviousValues, size_t sizeCurrentValues, double previousToCurrentWeights[][sizePreviousValues]);

// Backpropagation
void computeGradientToLayer(double expectedLayer[]);
void applyGradientToLayer();
double costFunction(double lastLayer[], double expextedValuesLayer[], size_t sizelastLayer);

#endif // NEURAL_NETWORK_H

