#include <vector>
#include "Utils.h"
#include "Neuron.h"


using namespace std;

Neuron::Neuron(vector<double>& weights, double bias) : weights(weights), bias(bias) {}

Neuron::Neuron() : weights({}), bias(0) {}

double Neuron::FeedForward(vector<double> inputs) {
	double total = Utils::DotProduct(weights, inputs) + bias;
	return Utils::sigmoid(total);
}

