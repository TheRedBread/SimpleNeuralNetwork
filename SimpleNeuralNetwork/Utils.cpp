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
