#include <math.h>
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace std;

class Utils {
public:

	static double sigmoid(double x) {
		return 1 / (1 + exp(-x));
	}
	static double DotProduct(vector<double> a, vector<double> b) {
		if (a.size() != b.size())
			throw invalid_argument("Vectors must have the same size");

		double result = 0;

		for (int i = 0; i < a.size(); i++) {
			result += a[i] * b[i];
		}

	}
};