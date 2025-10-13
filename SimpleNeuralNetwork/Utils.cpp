#include "Utils.h"

double Utils::DotProduct(const vector<double>& v1, const vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

double Utils::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Utils::deriv_sigmoid(double x) {
    double fx = sigmoid(x);
    return fx * (1 - fx);
}

double Utils::MSEloss(vector<double> y_true, vector<double> y_pred) {
   
    double result = 0;
    
    for (int i = 0; i < y_true.size(); i++) {
        result += pow(y_true[i] - y_pred[i], 2);
    }
    return (result) / y_true.size();


}

