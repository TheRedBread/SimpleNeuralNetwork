#include <vector>
#include <math.h>
#include "Neuron.h"
#include <random>
#include "Utils.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <iomanip>

std::random_device rd;                      
std::mt19937 gen(rd());                     
std::normal_distribution<> dist(0.0, 1.0); 

using namespace std;

// Weights
double w1 = dist(gen);
double w2 = dist(gen);
double w3 = dist(gen);
double w4 = dist(gen);
double w5 = dist(gen);
double w6 = dist(gen);

// biases
double b1 = dist(gen);  
double b2 = dist(gen);
double b3 = dist(gen);

NeuralNetwork::NeuralNetwork() {



}

double NeuralNetwork::FeedForward(vector<double> x) {
    double out_h1 = Utils::sigmoid(w1 * x[0] + w2 * x[1] + b1);
    double out_h2 = Utils::sigmoid(w3 * x[0] + w4 * x[1] + b2);

    double out_o1 = Utils::sigmoid(w5 * out_h1 + w6 * out_h2 + b3);

    return out_o1;
}

void NeuralNetwork::Train(vector<vector<double>> data, vector<double> all_y_trues) {
    double learn_rate = 0.1;
    int epochs = 1000; // number of loop times

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (size_t i = 0; i < data.size(); ++i) {
            std::vector<double> x = data[i];
            double y_true = all_y_trues[i];


            double sum_h1 = w1 * x[0] + w2 * x[1] + b1;
            double h1 = Utils::sigmoid(sum_h1);

            double sum_h2 = w3 * x[0] + w4 * x[1] + b2;
            double h2 = Utils::sigmoid(sum_h2);
            
            double sum_o1 = w5 * h1 + w6 * h2 + b3;
            double o1 = Utils::sigmoid(sum_o1);
            double y_pred = o1;

            // --- Calculate partial derivatives.
            // --- Naming: d_L_d_w1 represents "partial L / partial w1"

            double d_L_d_ypred = -2 * (y_true - y_pred);

            double d_ypred_d_w5 = h1 * Utils::deriv_sigmoid(sum_o1);
            double d_ypred_d_w6 = h2 * Utils::deriv_sigmoid(sum_o1);
            double d_ypred_d_b3 = Utils::deriv_sigmoid(sum_o1);

            double d_ypred_d_h1 = w5 * Utils::deriv_sigmoid(sum_o1);
            double d_ypred_d_h2 = w6 * Utils::deriv_sigmoid(sum_o1);

            // Neuron h1
            double d_h1_d_w1 = x[0] * Utils::deriv_sigmoid(sum_h1);
            double d_h1_d_w2 = x[1] * Utils::deriv_sigmoid(sum_h1);
            double d_h1_d_b1 = Utils::deriv_sigmoid(sum_h1);

            // Neuron h2
            double d_h2_d_w3 = x[0] * Utils::deriv_sigmoid(sum_h2);
            double d_h2_d_w4 = x[1] * Utils::deriv_sigmoid(sum_h2);
            double d_h2_d_b2 = Utils::deriv_sigmoid(sum_h2);

            // --- Aktualizacja wag i bias�w
            // Neuron h1
            w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
            w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
            b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;

            // Neuron h2
            w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
            w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
            b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;

            // Neuron o1
            w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5;
            w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6;
            b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3;

            if (epoch % 10 == 0) {
            
                vector<double> y_preds(data.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    y_preds[i] = FeedForward(data[i]); 
                }

                double loss = Utils::MSEloss(all_y_trues, y_preds);

                cout << "Epoch " << epoch << " loss: " << std::fixed << std::setprecision(3) << loss << std::endl;
            
            }


        }

    }


}

