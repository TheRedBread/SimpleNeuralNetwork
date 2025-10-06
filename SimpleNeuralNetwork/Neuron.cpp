#include <vector>
#include "Utils.cpp"

using namespace std;

class Neuron
{
public:
	vector<double> weights;
	double bias;

	Neuron(vector<double>& weights, double bias) : weights(weights), bias(bias) {}

	double FeedForward(vector<double> inputs) {
		double total = Utils::DotProduct(weights, inputs) + bias;
		return Utils::sigmoid(total);
	}

};

