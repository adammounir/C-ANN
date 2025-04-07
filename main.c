#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "neural_network.h"
#include "mnist.h"

#define BAR_WIDTH 50

// Generates a random number between -0.05 and 0.05

double randomnumber()
{
    return ((double)rand() / ((double) RAND_MAX)) * 0.1 - 0.05;
}

void saveModel(const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("[ERROR] Could not save the model to %s\n", filename);
        return;
    }

    fwrite(inputToHidden1, sizeof(double), numHidden1 * numInput, file);
    fwrite(hidden1ToHidden2, sizeof(double), numHidden2 * numHidden1, file);
    fwrite(hidden2ToOutput, sizeof(double), numOutput * numHidden2, file);

    fwrite(hidden1Bias, sizeof(double), numHidden1, file);
    fwrite(hidden2Bias, sizeof(double), numHidden2, file);
    fwrite(outputBias, sizeof(double), numOutput, file);

    fclose(file);
    printf("\n\033[1;32m[INFO] Model saved successfully to %s\033[0m\n", filename);
}

void loadModel(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("\n\033[1;31m[ERROR] Could not load the model from %s\033[0m\n", filename);
        return;
    }

    fread(inputToHidden1, sizeof(double), numHidden1 * numInput, file);
    fread(hidden1ToHidden2, sizeof(double), numHidden2 * numHidden1, file);
    fread(hidden2ToOutput, sizeof(double), numOutput * numHidden2, file);

    fread(hidden1Bias, sizeof(double), numHidden1, file);
    fread(hidden2Bias, sizeof(double), numHidden2, file);
    fread(outputBias, sizeof(double), numOutput, file);

    fclose(file);
    printf("\n\033[1;32m[INFO] Model loaded successfully from %s\033[0m\n", filename);
}

void printProgressBar(int current, int total, int epoch, int EPOCHS, time_t start)
{
    int progress = (int)((double)current / total * BAR_WIDTH);
    printf("\r\033[1;34m[TRAINING - EPOCH %d/%d]\033[0m [", epoch + 1, EPOCHS);
    for (int i = 0; i < BAR_WIDTH; i++)
    {
        if (i < progress) printf("=");
        else if (i == progress) printf(">");
        else printf(" ");
    }
    printf("] %d%%", (int)((double)current / total * 100));

    // Calculating ETA
    time_t now = time(NULL);
    double elapsed = difftime(now, start);
    double progressRatio = (double)(epoch * total + current) / (EPOCHS * total);
    double estimatedTotalTime = elapsed / progressRatio;
    double remainingTime = estimatedTotalTime - elapsed;
    int minutes = (int)(remainingTime / 60);
    int seconds = (int)((int)remainingTime % 60);

    printf(" | ETA: %02d:%02d", minutes, seconds);
    fflush(stdout);
}

void trainNetwork()
{
    load_mnist();

    assignRandomToVector(hidden1Bias, numHidden1);
    assignRandomToVector(hidden2Bias, numHidden2);
    assignRandomToVector(outputBias, numOutput);

    assignRandomToMatrix(numHidden1, numInput, inputToHidden1);
    assignRandomToMatrix(numHidden2, numHidden1, hidden1ToHidden2);
    assignRandomToMatrix(numOutput, numHidden2, hidden2ToOutput);

    const int EPOCHS = 1;
    time_t start = time(NULL);

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        printf("\n\033[1;33m[EPOCH %d/%d]\033[0m\n", epoch + 1, EPOCHS);
        for (int numImg = 0; numImg < NUM_TRAIN; numImg++)
        {
            for (int pix = 0; pix < SIZE; pix++)
                inputLayer[pix] = ((double)(train_image_char[numImg][pix])) / 255.0;

            for (int i = 0; i < numOutput; i++)
                expectedLayer[i] = (i == train_label[numImg]) ? 1.0 : 0.0;

            frontPropagation();
            softmax(outputLayer, numOutput);
            applyGradientToLayer();

            if ((numImg + 1) % 100 == 0 || numImg + 1 == NUM_TRAIN)
                printProgressBar(numImg + 1, NUM_TRAIN, epoch, EPOCHS, start);
        }
    }
    saveModel("model.txt");
    evaluateAccuracy();
}

int main()
{
    srand(time(NULL));

    printf("\033[1;36m===============================================================\n");
    printf("             NEURAL NETWORK - MNIST DIGIT RECOGNITION         \n");
    printf("===============================================================\033[0m\n");

    int choice;
    printf("\n\033[1;34m[MENU]\033[0m Choose an action:\n");
    printf("\033[1;32m 1 \033[0m- Train a new model\n");
    printf("\033[1;32m 2 \033[0m- Test using the saved model\n");
    printf("\033[1;32m 3 \033[0m- Quit\n");
    printf("\n\033[1;33mYour choice: \033[0m");
    scanf("%d", &choice);

    if (choice == 1)
    {
        trainNetwork();
    }
    else if (choice == 2)
    {
        loadModel("model.txt");
        evaluateAccuracy();
    }
    else if (choice == 3)
    {
        printf("\n\033[1;31m[EXIT] Program terminated.\033[0m\n");
        return 0;
    }
    else
    {
        printf("\n\033[1;31m[ERROR] Invalid choice.\033[0m\n");
    }

    return 0;
}

