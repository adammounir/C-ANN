/*-----------------------------------------------------------------------------*/
//                              INCLUDES 

#include "neural_network.h"

#define LEARNING_RATE 0.05

// ∂C/∂wl = (∂il/∂wl) * (∂ol1/∂il) * (∂C/∂ol1)
void computeGradientToLayer(double expectedLayer[])
{
    //Initailzes the secondAndThirdPartArray 
    double secondAndThirdPartArray[numOutput];

    // OUTPUT LAYER GRADIENT 
    for(int nthoutput=0; nthoutput<numOutput; nthoutput++)
    {
        double secondAndThirdPart = outputLayer[nthoutput]*(1 - outputLayer[nthoutput]) * 2*(outputLayer[nthoutput] - expectedLayer[nthoutput]);

        gradBiasesOuputLayer[nthoutput] = secondAndThirdPart;
        secondAndThirdPartArray[nthoutput] = secondAndThirdPart;

        for (int nthhidden2=0; nthhidden2<numHidden2; nthhidden2++)
        {
            double gradWeight = hiddenLayer2[nthhidden2];
            gradWeight *= secondAndThirdPart;
            gradWeightsHidden2ToOutput[nthoutput][nthhidden2] = gradWeight;
        }
    }

    //1st HIDDEN LAYER GRADIENT (first hidden layer)
    for(int nthhidden1=0; nthhidden1<numHidden1; nthhidden1++)
    {
        double secondPart = hiddenLayer1[nthhidden1]*(1 - hiddenLayer1[nthhidden1]);
        double thirdPart = 0;
        for (int nthhidden2=0; nthhidden2<numHidden2; nthhidden2++)
        {
            thirdPart += hidden1ToHidden2[nthhidden2][nthhidden1] * secondAndThirdPartArray[nthhidden2];
        }
        gradBiasesHiddenLayer1[nthhidden1] = secondPart * thirdPart;

        for(int nthinput=0; nthinput<numInput; nthinput++)
        {
            double firstPart = inputLayer[nthinput];
            gradWeightsInputToHidden1[nthhidden1][nthinput] = firstPart * secondPart * thirdPart;
        }
    }

    //2nd HIDDEN LAYER GRADIENT (second hidden layer)
    for(int nthhidden2=0; nthhidden2<numHidden2; nthhidden2++)
    {
        double secondPart = hiddenLayer2[nthhidden2]*(1 - hiddenLayer2[nthhidden2]);
        double thirdPart = 0;
        for (int nthoutput=0; nthoutput<numOutput; nthoutput++)
        {
            thirdPart += hidden2ToOutput[nthoutput][nthhidden2] * secondAndThirdPartArray[nthoutput];
        }
        gradBiasesHiddenLayer2[nthhidden2] = secondPart * thirdPart;

        for(int nthhidden1=0; nthhidden1<numHidden1; nthhidden1++)
        {
            double firstPart = hiddenLayer1[nthhidden1];
            gradWeightsHidden1ToHidden2[nthhidden2][nthhidden1] = firstPart * secondPart * thirdPart;
        }
    }
}


void applyGradientToLayer()
{
    computeGradientToLayer(expectedLayer);

    for (int nthoutput=0; nthoutput<numOutput ; nthoutput++)
    {
        outputBias[nthoutput] -= LEARNING_RATE * gradBiasesOuputLayer[nthoutput];

        for (int nthhidden2=0; nthhidden2<numHidden2; nthhidden2++)
        {
            hidden2ToOutput[nthoutput][nthhidden2] -= LEARNING_RATE * gradWeightsHidden2ToOutput[nthoutput][nthhidden2];
            hidden2Bias[nthhidden2] -= LEARNING_RATE * gradBiasesHiddenLayer2[nthhidden2];

            for (int nthhidden1=0; nthhidden1<numHidden1 ; nthhidden1++)
            {
                hidden1ToHidden2[nthhidden2][nthhidden1] -= LEARNING_RATE * gradWeightsHidden1ToHidden2[nthhidden2][nthhidden1];
                hidden1Bias[nthhidden1] -= LEARNING_RATE * gradBiasesHiddenLayer1[nthhidden1];

                for (int nthinput=0 ; nthinput<numInput ; nthinput++)
                {
                    inputToHidden1[nthhidden1][nthinput] -= LEARNING_RATE * gradWeightsInputToHidden1[nthhidden1][nthinput];
                }
            }
        }
    }
}

