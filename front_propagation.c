//                  INCLUDE LIBRARY
#include "neural_network.h"

// Applies the ReLU function

double relu(double x)
{
    return x > 0 ? x : 0;
}

// Assigns random values to a matrix
void assignRandomToMatrix(size_t currentdim, size_t prevdim, double layer[currentdim][prevdim])
{
    for(size_t i=0; i<currentdim; i++)
    {
       for(size_t j=0; j<prevdim; j++)
       {
            layer[i][j] = randomnumber();
       }
    }
}

// Assigns random values to a vector
void assignRandomToVector(double layer[], int layerSize)
{
    for(int i=0; i<layerSize; i++)
    {
        double temp = randomnumber();
        layer[i] = temp;
    }
}

// Computes current layer values according to the previous one
void layerValues(double previousValues[], double currentValues[], double currentBias[], size_t sizePreviousValues, size_t sizeCurrentValues,double previousToCurrentWeights[][sizePreviousValues])
{
    for(size_t i=0; i<sizeCurrentValues; i++)
    {
        currentValues[i] = 0;
        for(size_t j=0; j<sizePreviousValues; j++)
        {
            currentValues[i] += previousValues[j] * previousToCurrentWeights[i][j];
        }
        currentValues[i] += currentBias[i];
        currentValues[i] = relu(currentValues[i]);  // Using ReLU instead of Sigmoid
    }
}


// Function that makes the front propagation 
void frontPropagation()
{
    layerValues(inputLayer, hiddenLayer1, hidden1Bias, numInput, numHidden1, inputToHidden1);
    layerValues(hiddenLayer1, hiddenLayer2, hidden2Bias, numHidden1, numHidden2, hidden1ToHidden2);
    layerValues(hiddenLayer2, outputLayer, outputBias, numHidden2, numOutput, hidden2ToOutput);
}

