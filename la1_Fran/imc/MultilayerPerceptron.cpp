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
MultilayerPerceptron::MultilayerPerceptron(double ETA, double MU)
{
	//Made by me
	eta = ETA;
	mu = MU;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vector containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, std::vector<int> npl) {
	//Changed by me
	nOfLayers = nl;
	layers = std::vector<Layer>(nOfLayers);

	//Initialize layer 0 or input layer
	layers[0].nOfNeurons = npl[0];
	layers[0].neurons = std::vector<Neuron>(layers[0].nOfNeurons);
	for ( std::vector<Neuron>::iterator it = layers[0].neurons.begin(); it != layers[0].neurons.end(); it++ )
	{
		it->out = 0;
		it->delta = 0;
		it->w = std::vector<double>();
		it->deltaW = std::vector<double>();
		it->lastDeltaW = std::vector<double>();
		it->wCopy = std::vector<double>();
	}

	//Hidden and output layers
	for(int i = 1; i < nOfLayers; i++){
		layers[i].nOfNeurons = npl[i];
		layers[i].neurons = std::vector<Neuron>(layers[i].nOfNeurons);

		//Initialize Neuron's data
		//n_weights = layers[i-1].nOfNeurons + 1
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			layers[i].neurons[j].out = 0.0;
			layers[i].neurons[j].delta = 0.0;
			layers[i].neurons[j].w = std::vector<double>(npl[i-1]+1, 0.0);	//Neurons number of previous layer + 1 (bias)
			layers[i].neurons[j].deltaW = std::vector<double>(npl[i-1]+1, 0.0);;
			layers[i].neurons[j].lastDeltaW = std::vector<double>(npl[i-1]+1, 0.0);;
			layers[i].neurons[j].wCopy = std::vector<double>(npl[i-1]+1, 0.0);
		}
	}

	return nOfLayers;
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
	//Made by me
	for ( std::vector<Layer>::iterator itLayer = layers.begin(); itLayer != layers.end(); itLayer++ )
	{
		for ( std::vector<Neuron>::iterator itNeuron = itLayer->neurons.begin(); itNeuron != itLayer->neurons.end(); itNeuron++ )
		{
			itNeuron->out = 0.0;
			itNeuron->delta = 0.0;
			itNeuron->w.clear();
			itNeuron->deltaW.clear();
			itNeuron->lastDeltaW.clear();
			itNeuron->wCopy.clear();
		}

		itLayer->nOfNeurons = 0.0;
		itLayer->neurons.clear();
	}

	nOfLayers = 0;
	layers.clear();
}

// ------------------------------
// Feel all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
	//Made by me
	for ( std::vector<Layer>::iterator itLayer = layers.begin(); itLayer != layers.end(); itLayer++ )
	{
		for ( std::vector<Neuron>::iterator itNeuron = itLayer->neurons.begin(); itNeuron != itLayer->neurons.end(); itNeuron++ )
		{
			for ( std::vector<double>::iterator itWeight = itNeuron->w.begin(); itWeight != itNeuron->w.end(); itWeight++ )
			{
				*itWeight = randomDouble(-1.0,1.0);
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
void MultilayerPerceptron::getOutputs(std::vector <double> output)
{
	//Made by me
	int h = nOfLayers-1;	//Last or output layer

	for ( vector<Neuron>::iterator it = layers[h].neurons.begin(); it != layers[h].neurons.end(); it++ )
	{
		output.push_back( it->out );
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
	//Made by me
	for ( std::vector<Layer>::iterator itLayer = layers.begin(); itLayer != layers.end(); itLayer++  )
	{
		for ( std::vector<Neuron>::iterator itNeuron = itLayer->neurons.begin(); itNeuron != itLayer->neurons.end(); itNeuron++ )
		{
			itNeuron->wCopy = itNeuron->w;
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
	//Made by me
	for ( std::vector<Layer>::iterator itLayer = layers.begin(); itLayer != layers.end(); itLayer++  )
	{
		for ( std::vector<Neuron>::iterator itNeuron = itLayer->neurons.begin(); itNeuron != itLayer->neurons.end(); itNeuron++ )
		{
			itNeuron->w = itNeuron->wCopy;
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
	//Made by me
	//Start since second Layer because first are inputs
	//Layer = h, Neuron = j, Weight = i
	for ( int h = 1; h < nOfLayers; h++ )
	{
		for ( int j = 0; j < layers[h].nOfNeurons; j++ )
		{
			//Go through weights until the next to last because the last is the bias
			double net = 0.0; 
			for( int i = 0; i < (layers[h-1].nOfNeurons); i++ )
			{
				//Net += CurrentNeuron.weight[i] * PreviousNeuron[i]
				net += layers[h].neurons[j].w[i] * (layers[h-1].neurons[i].out) ;
			}

			//Add bias value
			net += *(layers[h].neurons[j].w.end());

			//Apply the activation (sigmoide) function
			layers[h].neurons[j].out = 1 / ( 1 + exp(-net) );
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(std::vector<double> target) {
	//Made by me
	//FIXME: check this function
	int outputLayer = nOfLayers-1;
	int nNeuronsOutputLayer = layers[outputLayer].nOfNeurons;
	double error = 0.0;

	for ( int i = 0; i < nNeuronsOutputLayer; i++ )
	{
		error += std::pow( layers[outputLayer].neurons[i].out - target[i], 2 );
	}

	return error / (double)nNeuronsOutputLayer;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(std::vector<double> target) {
	//Made by me
	//For each output neuron
	int lastLayer = nOfLayers-1;

	for ( int j = 0; j < layers[lastLayer].nOfNeurons; j++ )
	{
		//It does the formula more visual
		double dj = target[j];
		double outHj = layers[lastLayer].neurons[j].out;

		//-(resulObjetivo[j] - salida[j]) * g'(net^H_j) //La derivada es: salida[j]*(1-salida[j])
		layers[lastLayer].neurons[j].delta = -(dj - outHj) * (outHj * ( 1 - outHj ) );
	}

	//For each layer
	for ( int h = nOfLayers-2; h >= 0; h-- )
	{
		for ( int j = 0; j < layers[h].nOfNeurons; j++ )
		{
			double summation = 0.0;
			for( int i = 0; i < layers[h+1].nOfNeurons; i++ )
			{
				//It does the formula more visual
				double weightH1ij = layers[h+1].neurons[i].w[j];
				double deltaH1i = layers[h+1].neurons[i].delta;

				summation += weightH1ij * deltaH1i;
			}
			//It does the formula more visual
			double outhj = layers[h].neurons[j].out;

			layers[h].neurons[j].delta = summation * outhj * (1 - outhj);
		}
	}
}


// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	//Made by me
	//For each layer
	for ( int h = 1; h < nOfLayers; h++ )
	{
		//For each neuron of layer h
		for ( int j = 0; j < layers[h].nOfNeurons; j++ )
		{
			//For each neuron of layer h-1
			for ( int i = 0; i < layers[h-1].nOfNeurons; i++ )
			{
				//It does the formula more visual
				double deltaWhji = layers[h].neurons[j].deltaW[i];
				double deltahj = layers[h].neurons[j].delta;
				double outh1i = layers[h-1].neurons[i].out;

				layers[h].neurons[j].deltaW[i] =  deltaWhji + deltahj * outh1i;
				//FIXME: check this vvvvvvv because if not, on weightAdjustment the lastDeltaW is 0
				layers[h].neurons[j].lastDeltaW[i] = layers[h].neurons[j].deltaW[i];
			}

			//Bias
			//It does the formula more visual
			int biasPosition = layers[h-1].nOfNeurons;
			double deltaWhjBias = layers[h].neurons[j].deltaW[biasPosition];
			double deltahj = layers[h].neurons[j].delta;

			layers[h].neurons[j].deltaW[biasPosition] = deltaWhjBias + deltahj * 1;
			//FIXME: check this vvvvvvv because if not, on weightAdjustment the lastDeltaW is 0
			layers[h].neurons[j].lastDeltaW[biasPosition] = layers[h].neurons[j].deltaW[biasPosition];
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
	//Made by me. Use slide 13 of lab pdf
	//For each layer
	for( int h = 1; h < nOfLayers; h++)
	{
		//For each neuron of layer h
		for ( int j = 0; j < layers[h].nOfNeurons; j++)
		{
			//For each neuron of layer h-1
			for ( int i = 0; i < layers[h-1].nOfNeurons; i++ )
			{
				//It does the formula more visual
				double whji = layers[h].neurons[j].w[i];
				double deltaWhji = layers[h].neurons[j].deltaW[i];
				double lastDeltaWhji = layers[h].neurons[j].lastDeltaW[i];

				layers[h].neurons[j].w[i] = whji - eta * deltaWhji - mu * ( eta * lastDeltaWhji );
				//FIXME: lastDeltaW here or above?vvvvvvvvvvvv
				layers[h].neurons[j].lastDeltaW[i] = eta * deltaWhji;
			}

			//Bias
			//It does the formula more visual
			int positionBias = layers[h-1].nOfNeurons;
			double whjBias = layers[h].neurons[j].w[positionBias];
			double deltaWhjBias = layers[h].neurons[j].deltaW[positionBias];
			double lastDeltaWhjBias = layers[h].neurons[j].lastDeltaW[positionBias];

			layers[h].neurons[j].w[positionBias] = whjBias - eta * deltaWhjBias - mu * ( eta * lastDeltaWhjBias );
			//FIXME: lastDeltaW here or above?vvvvvvvvvvvv
			layers[h].neurons[j].lastDeltaW[positionBias] = eta * deltaWhjBias;
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
	std::cout<< "WARNING: the neural network is shown from top to botton. Top is inputs and bottom is outputs.\n";
	for ( int h = 0; h < nOfLayers; h++)
	{
		std::cout << "//////////////////\n";
		std::cout << "Capa " << h <<":";

		for ( int j = 0; j < layers[h].nOfNeurons; j++)
		{
			std::cout << "+++++++++++++++++\n";
			std::cout << "Neurona " << j <<":";

			for ( int i = 0; i < layers[h].neurons[j].w.size(); i++)
			{
				std::cout << "-----------------\n";
				std::cout << "Pesos " << i <<":";
				
				for (double weight: layers[h].neurons[j].w){
	    			std::cout << weight << ' ';
				}
				std::cout << std::endl;
				std::cout << "-----------------\n";
			}
			std::cout << "+++++++++++++++++\n";
		}
		std::cout << "//////////////////\n";
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(std::vector<double> input, std::vector<double> target) {
	// The 5 steps of Backpropagation (Slide 8 lab pdf)
	// Changes will be applied for each pattern
	for ( std::vector<Layer>::iterator itLayer = layers.begin(); itLayer != layers.end(); itLayer++ )
	{
		for ( std::vector<Neuron>::iterator itNeuron = itLayer->neurons.begin(); itNeuron != itLayer->neurons.end(); itNeuron++)
		{
			for ( std::vector<double>::iterator itDeltaW = itNeuron->deltaW.begin(); itDeltaW != itNeuron->deltaW.end(); itDeltaW++)
			{
				*itDeltaW = 0.0;
			}
		}
	}

	//Feed inputs
	feedInputs(input);

	//Forward propagation
	forwardPropagate();

	//Error backpropagation
	backpropagateError(target);

	//Obtain the weight update
	accumulateChange();

	//Apply the calculated update
	weightAdjustment();
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
	//Changed by me
	//FIXME: check if it is correct
	//TODO: check if it is correct
	double error = 0.0;
/*	PREVIOUS function
	for ( int i = 0; i < testDataset->nOfPatterns; i++ )
	{
		for ( int j = 0; j < layers[0].nOfNeurons; j++ )
		{
			layers[0].neurons[j].out = testDataset->inputs[i][j];
		}

		forwardPropagate();

		error += obtainError(testDataset->outputs[i]);
	}
*/	

	for( int i = 0; i < testDataset->nOfPatterns; i++ )
	{
		//Feed inputs
		feedInputs(testDataset->inputs[i]);

		//Forward propagation
		forwardPropagate();

		//Error backpropagation
		backpropagateError(testDataset->outputs[i]);

		error += obtainError(testDataset->outputs[i]);
	}

	return error / testDataset->nOfPatterns;
}


// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset* pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	std::vector<double> obtained = std::vector<double>(numSalidas);	//Changed by me
	
	std::cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nOfPatterns; i++){

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);
		
		std::cout << i;

		for (j = 0; j < numSalidas; j++)
			std::cout << "," << obtained[j];
		std::cout << endl;

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
			std::cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}


		countTrain++;

		std::cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while ( countTrain<maxiter );

	std::cout << "NETWORK WEIGHTS" << endl;
	std::cout << "===============" << endl;
	printNetwork();

	std::cout << "Desired output Vs Obtained output (test)" << endl;
	std::cout << "=========================================" << endl;
	for(int i=0; i<pDatosTest->nOfPatterns; i++){
		std::vector <double> prediction = std::vector<double>(pDatosTest->nOfOutputs);	//Changed by me

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<pDatosTest->nOfOutputs; j++)
			std::cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
		std::cout << endl;
		prediction.clear();	//Changed by me

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
	npl.clear();	//Changed by me

	return true;
}
