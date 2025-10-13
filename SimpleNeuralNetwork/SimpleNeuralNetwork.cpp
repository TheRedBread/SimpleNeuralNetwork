#include <iostream>
#include <vector>
#include "NeuralNetwork.h"
#include "Utils.h"

using namespace std;



int main()
{
    vector<std::vector<double>> data = {
        {-2, -1},   // Alice
        {25, 6},    // Bob
        {17, 4},    // Charlie
        {-15, -6}   // Diana
    };
    vector<double> all_y_trues = {
        1,  // Alice
        0,  // Bob
        0,  // Charlie
        1   // Diana
    };

    NeuralNetwork network;
    network.Train(data, all_y_trues);

    return 0;

}

