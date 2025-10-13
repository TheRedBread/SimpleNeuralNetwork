#pragma once
#include <vector>
#include "Neuron.h" 
using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork();
    double FeedForward(vector<double> x);
    void Train(vector<vector<double>> data, vector<double> all_y_trues);
};