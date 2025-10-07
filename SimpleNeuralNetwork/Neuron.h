#pragma once
#include <vector>
using namespace std;

class Neuron
{
public:
	vector<double> weights;
	double bias;

	Neuron(vector<double>& weights, double bias);
	Neuron();

	double FeedForward(vector<double> inputs);
};
