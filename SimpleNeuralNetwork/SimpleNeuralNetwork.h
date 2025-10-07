#pragma once
#include <vector>
#include "Neuron.h"
using namespace std;

class NeuralNetwork {
public:
    /*
    A neural network with :
        -2 inputs
        - a hidden layer with 2 neurons(h1, h2)
        - an output layer with 1 neuron(o1)
    Each neuron has the same weights and bias :
        -w = [0, 1]
        - b = 0
    */

    vector<double> weights;
    double bias;

    Neuron h1;
    Neuron h2;
    Neuron o1;

    NeuralNetwork();
    double FeedForward(vector<double> x);
};
