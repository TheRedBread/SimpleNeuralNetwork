#include <vector>
#include "Neuron.h"
#include "SimpleNeuralNetwork.h"

using namespace std;

NeuralNetwork::NeuralNetwork() {
    weights = { 0, 1 };
    bias = 0;

    h1 = Neuron(weights, bias);
    h2 = Neuron(weights, bias);
    o1 = Neuron(weights, bias);
}

double NeuralNetwork::FeedForward(vector<double> x) {
    double out_h1 = h1.FeedForward(x);
    double out_h2 = h2.FeedForward(x);

    double out_o1 = o1.FeedForward({ out_h1, out_h2 });
    return out_o1;
}