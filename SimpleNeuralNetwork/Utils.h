#pragma once
#include <vector>
#include <cmath>
using namespace std;

class Utils {
public:
    static double DotProduct(const vector<double>& v1, const vector<double>& v2);
    static double sigmoid(double x);
    static double MSEloss(vector<double> y_true, vector<double> y_pred);
};
