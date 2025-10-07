#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

using namespace std;



int main()
{
	NeuralNetwork network = NeuralNetwork();

	vector<double> x = { 2, 3 };

	cout << network.FeedForward(x);

}

