#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()

#include "util.h"
//Added by me
#include <algorithm>    //sort
#include <limits>       //infinity

using namespace std;
using namespace util;


// ------------------------------
// Obtain an integer random number in the range [Low,High]
int util::randomInt(int Low, int High)
{
	return rand() % (High-Low+1) + Low;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double util::randomDouble(double Low, double High)
{
	return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset *util::readData(const char *fileName)
{

    ifstream myFile(fileName); // Create an input stream

    if (!myFile.is_open())
    {
        cout << "ERROR: I cannot open the file " << fileName << endl;
        return NULL;
    }

    Dataset *dataset = new Dataset;
    if (dataset == NULL)
        return NULL;

    string line;
    int i, j;

    if (myFile.good())
    {
        getline(myFile, line); // Read a line
        istringstream iss(line);
        iss >> dataset->nOfInputs;
        iss >> dataset->nOfOutputs;
        iss >> dataset->nOfPatterns;
    }
    (dataset->inputs).resize(dataset->nOfPatterns);    //Changed by me
    (dataset->outputs).resize(dataset->nOfPatterns);   //Changed by me

    for (i = 0; i < dataset->nOfPatterns; i++)
    {
        dataset->inputs[i].resize(dataset->nOfInputs);  //Changed by me
        dataset->outputs[i].resize(dataset->nOfOutputs);    //Changed by me
    }

    i = 0;
    while (myFile.good())
    {
        getline(myFile, line); // Read a line
        if (!line.empty())
        {
            istringstream iss(line);
            for (j = 0; j < dataset->nOfInputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->inputs[i][j] = value;
            }
            for (j = 0; j < dataset->nOfOutputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->outputs[i][j] = value;
            }
            i++;
        }
    }

    myFile.close();

    return dataset;
}


// ------------------------------
// Print the dataset
void util::printDataset(Dataset *dataset, int len)
{
    if (len == 0)
        len = dataset->nOfPatterns;

    for (int i = 0; i < len; i++)
    {
        cout << "P" << i << ":" << endl;
        for (int j = 0; j < dataset->nOfInputs; j++)
        {
            cout << dataset->inputs[i][j] << ",";
        }

        for (int j = 0; j < dataset->nOfOutputs; j++)
        {
            cout << dataset->outputs[i][j] << ",";
        }
        cout << endl;
    }
}

// ------------------------------
// Transform an scalar x by scaling it to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). 
double util::minMaxScaler(double x, double minAllowed, double maxAllowed, double minData, double maxData)
{
    //Made by me
    return minAllowed + (( x - minData ) * (maxAllowed-minAllowed)) / (maxData - minData);
}

// ------------------------------
// Scale the dataset inputs to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). 
void util::minMaxScalerDataSetInputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                     std::vector<double> minData, std::vector<double> maxData)
{
    //Made by me
    //Go through by column
    for (int i = 0; i < dataset->nOfInputs; i++){
        for(int j = 0; j < dataset->nOfPatterns; j++){
            ((dataset->inputs)[j])[i] = minMaxScaler( ((dataset->inputs)[j])[i], minAllowed, maxAllowed, minData[i], maxData[i] );
        }
    }
}

// ------------------------------
// Scale the dataset output vector to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). Only for regression problems. 
void util::minMaxScalerDataSetOutputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                      double minData, double maxData)
{
    //Made by me
    //Go through by column
    for (int i = 0; i < dataset->nOfOutputs; i++){
        for(int j = 0; j < dataset->nOfPatterns; j++){
            dataset->outputs[j][i] = minMaxScaler( ((dataset->inputs)[j])[i], minAllowed, maxAllowed, minData, maxData );
        }
    }
}

// ------------------------------
// Get a vector of minimum values of the dataset inputs
std::vector<double> util::minDatasetInputs(Dataset *dataset)
{
    //Made by me
    std::vector<double> min = std::vector<double>(dataset->nOfInputs,std::numeric_limits<double>::infinity());

    for (int i = 0; i < dataset->nOfInputs; i++){
        for(int j = 0; j < dataset->nOfPatterns; j++){
            if ( min[i] > dataset->inputs[j][i]){
                min[i] = dataset->inputs[j][i];
            }
        }
    }

    return min;
}

// ------------------------------
// Get a vector of maximum values of the dataset inputs
std::vector<double> util::maxDatasetInputs(Dataset *dataset)
{
    //Made by me
    std::vector<double> max = std::vector<double>(dataset->nOfInputs,-std::numeric_limits<double>::infinity());

    for (int i = 0; i < dataset->nOfInputs; i++){
        for(int j = 0; j < dataset->nOfPatterns; j++){
            if ( max[i] < dataset->inputs[j][i]){
                max[i] = dataset->inputs[j][i];
            }
        }
    }

    return max;
}

// ------------------------------
// Get the minimum value of the dataset outputs
double util::minDatasetOutputs(Dataset *dataset)
{
    //Made by me
    std::vector<double> min = std::vector<double>(dataset->nOfOutputs,std::numeric_limits<double>::infinity());

    for (int i = 0; i < dataset->nOfOutputs; i++){
        for(int j = 0; j < dataset->nOfPatterns; j++){
            if ( min[i] > dataset->outputs[j][i]){
                min[i] = dataset->outputs[j][i];
            }
        }
    }

    return *std::min_element( min.begin(), min.end() );
}

// ------------------------------
 // Get the maximum value of the dataset outputs
double util::maxDatasetOutputs(Dataset *dataset)
{
    //Made by me
    std::vector<double> max = std::vector<double>(dataset->nOfOutputs,-std::numeric_limits<double>::infinity());

    for (int i = 0; i < dataset->nOfOutputs; i++){
        for(int j = 0; j < dataset->nOfPatterns; j++){
            if ( max[i] < dataset->outputs[j][i]){
                max[i] = dataset->outputs[j][i];
            }
        }
    }

    return *std::max_element( max.begin(), max.end() );
}