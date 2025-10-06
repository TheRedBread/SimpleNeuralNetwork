#include <iostream>
#include <vector>
#include "Neuron.cpp"

using namespace std;



int main()
{
	vector<double> weights = {0, 1};
	double bias = 4;
	Neuron n = Neuron(weights, bias);

	vector<double> x = { 2, 3 };

	cout << n.FeedForward(x);

}

