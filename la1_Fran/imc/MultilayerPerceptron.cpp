/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <math.h>
//Added by me
#include <vector>

using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	//Made by me
	nOfLayers = 3;	//Inputs, 1 hidden, outputs
	layers.reserve(3);

	for (int i = 0; i < 2; i++)
	{
		layers[i].nOfNeurons = 5;
		layers[i].neurons.reserve(5);

		for ( int j = 0; i < 5; j++ )
		{
			layers[i].neurons[j].out = 0;
			layers[i].neurons[j].delta = 0;
			layers[i].neurons[j].w = std::vector<double>(6, 0.0);
			layers[i].neurons[j].deltaW = std::vector<double>(6, 0.0);
			layers[i].neurons[j].lastDeltaW = std::vector<double>(6, 0.0);
			layers[i].neurons[j].wCopy = std::vector<double>(6, 0.0);
		}
	}
	//Last layer (output layer)
	layers[2].nOfNeurons = 1;
	layers[2].neurons.reserve(1);

	layers[2].neurons[0].out = 0;
	layers[2].neurons[0].delta = 0;
	layers[2].neurons[0].w = std::vector<double>(5, 0.0);
	layers[2].neurons[0].deltaW = std::vector<double>(5, 0.0);
	layers[2].neurons[0].lastDeltaW = std::vector<double>(5, 0.0);
	layers[2].neurons[0].wCopy = std::vector<double>(5, 0.0);
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, std::vector<int> npl) {
	//Changed by me
	nOfLayers = nl;
	layers = std::vector<Layer>(nl);

	//Input and hidden layers
	for(int i = 0; i < (nl-1); i++){
		Layer tempLayer;
		tempLayer.nOfNeurons = npl[i];
		tempLayer.neurons = std::vector<Neuron>(npl[i]);

		//Initialize Neuron's data
		for (int j = 0; j < npl[i]; j++)
		{
			tempLayer.neurons[j].out = 0.0;
			tempLayer.neurons[j].delta = 0.0;
			tempLayer.neurons[j].w = std::vector<double>(npl[i], 0.0);
			tempLayer.neurons[j].deltaW = std::vector<double>(npl[i], 0.0);;
			tempLayer.neurons[j].lastDeltaW = std::vector<double>(npl[i], 0.0);;
			tempLayer.neurons[j].wCopy = std::vector<double>(npl[i], 0.0);
		}

		layers[i] = tempLayer;
	}
	// Output layer (1 neuron)
	Layer tempLayer;
	tempLayer.nOfNeurons = 1;
	tempLayer.neurons = std::vector<Neuron>(1);
	tempLayer.neurons[0].out = 0.0;
	tempLayer.neurons[0].delta = 0.0;
	tempLayer.neurons[0].w = std::vector<double>(npl[nl-2], 0.0);
	tempLayer.neurons[0].deltaW = std::vector<double>(npl[nl-2], 0.0);;
	tempLayer.neurons[0].lastDeltaW = std::vector<double>(npl[nl-2], 0.0);;
	tempLayer.neurons[0].wCopy = std::vector<double>(npl[nl-2], 0.0);
	layers[nl-1] = tempLayer;
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {

}

// ------------------------------
// Feel all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
	//Made by me
	for ( int h = 0; h < nOfLayers; h++ )
	{
		for ( int j = 0; j < layers[h].nOfNeurons; j++ )
		{
			for ( int i = 0; i < layers[h].neurons[j].w.size(); i++ )
			{
				layers[h].neurons[j].w[i] = randomDouble(-1.0,1.0);
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(std::vector<double> input) {
	//Made by me
	int h = 0;	//First or input layer

	for ( int j = 0; j < layers[h].nOfNeurons; j++ )
	{
		layers[h].neurons[j].out = input[j];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{
	//Made by me
	int h = nOfLayers-1;	//Last or output layer

	for ( int j = 0; j < layers[h].nOfNeurons; j++)
	{
		
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {

}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {

}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
	
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double* target) {
	return -1;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double* target) {
	
}


// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {

}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {


}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(std::vector<double> input, std::vector<double> target) {

}

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset* trainDataset) {
	int i;
	for(i=0; i<trainDataset->nOfPatterns; i++){
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset* testDataset) {
	return -1.0;
}


// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset* pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * obtained = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nOfPatterns; i++){

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);
		
		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;

	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset * trainDataset, Dataset * pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving;
	double testError = 0;

	// Learning
	do {

		trainOnline(trainDataset);
		double trainError = test(trainDataset);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}


		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while ( countTrain<maxiter );

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<pDatosTest->nOfPatterns; i++){
		double* prediction = new double[pDatosTest->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<pDatosTest->nOfOutputs; j++)
			cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	testError = test(pDatosTest);
	*errorTest=testError;
	*errorTrain=minTrainError;

}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	std::vector<int> npl;	//Changed by me

	// Read number of layers
	f >> nl;

	npl = std::vector<int>(nl);	//Changed by me

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] &npl;	//Changed by me

	return true;
}