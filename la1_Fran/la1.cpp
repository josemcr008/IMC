//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"
//Added by me
#include <vector>

using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Process arguments of the command line
    bool tflag = 0, Tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0, wflag = 0, pflag = 0, sflag = 0;    //Changed by me
    char *tvalue = NULL, *Tvalue = NULL, *wvalue = NULL;    //Changed by me
    int c, ivalue, lvalue, hvalue;
    float evalue = 0.0, mvalue = 0.0;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
//CODIGO EN FICHERO ARGUMENTOS.TXT

    //Added by me
    //Check if there are missing arguments and set some argument to default
/*  if ( !tflag)
    {
        fprintf(stderr, "Falta el archivo de entrenamiento.\n");
        return EXIT_FAILURE;
    }*/

    if ( !Tflag )
        Tvalue = tvalue;

    if ( !iflag )
        ivalue = 1000;
    
    if ( !lflag )
        lvalue = 1;
    
    if ( !hflag )
        hvalue = 5;
    
    if ( !eflag )
        evalue = 0.1;
    
    if ( !mflag )
        mvalue = 0.9;
    ///////////////////////////////////////////////////////////////////
tvalue = "/home/fran/Desktop/University/4Curso/1Cuatrimestre/IMC/Practica/Practica1/skeletonLA1IMC/la1/train_cropyield.dat";
Tvalue = "/home/fran/Desktop/University/4Curso/1Cuatrimestre/IMC/Practica/Practica1/skeletonLA1IMC/la1/test_cropyield.dat";

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp(evalue, mvalue);

        // Parameters of the mlp. For example, mlp.eta = value;
    	int iterations = ivalue; // This should be corrected    //Changed by me

        // Read training and test data: call to util::readData(...)
    	Dataset * trainDataset = util::readData(tvalue); // This should be corrected    //Changed by me
    	Dataset * testDataset = util::readData(Tvalue); // This should be corrected     //Changed by me

        // Normalization
        sflag = true;
        if ( sflag ){
            // Normalization of TRAIN dataset with Scalling
            std::vector<double> minInValues = minDatasetInputs(trainDataset);
            std::vector<double> maxInValues = maxDatasetInputs(trainDataset);
            double minOutValue = minDatasetOutputs(trainDataset);
            double maxOutValue = maxDatasetOutputs(trainDataset);

            minMaxScalerDataSetInputs(trainDataset, -1.0, 1.0, minInValues, maxInValues);
            minMaxScalerDataSetOutputs(trainDataset, -1.0, 1.0, minOutValue, maxOutValue);

            minMaxScalerDataSetInputs(testDataset, -1.0, 1.0, minInValues, maxInValues);
            minMaxScalerDataSetOutputs(testDataset, -1.0, 1.0, minOutValue, maxOutValue);
        }

        // Initialize topology vector
    	int layers= lvalue+2; // This should be corrected //Changed by me
    	std::vector<int> topology= std::vector<int>(layers, hvalue); // This should be corrected //Changed by me
        //Added by me //First and last layer (output layer)
        topology[0] = trainDataset->nOfInputs;
        topology[layers-1] = trainDataset->nOfOutputs;

        // Initialize the network using the topology vector
        mlp.initialize(layers,topology);


        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        std::vector<double> testErrors = std::vector<double>(5);    //Changed by me
        std::vector<double> trainErrors = std::vector<double>(5);   //Changed by me
        double bestTestError = 1;
        for(int i=0; i<5; i++){
            std::cout << "**********" << endl;
            std::cout << "SEED " << seeds[i] << endl;
            std::cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset,testDataset,iterations,&(trainErrors[i]),&(testErrors[i]));
            std::cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        std::cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;
        
        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        //TODO: check if it is correct
        for ( int i = 0; i < trainErrors.size(); i++){
            averageTrainError += trainErrors[i];
        }
        averageTrainError /= trainErrors.size();
        stdTrainError = sqrt(averageTrainError);

        for ( int i = 0; i < testErrors.size(); i++){
            averageTestError += testErrors[i];
        }
        averageTestError /= testErrors.size();
        stdTestError = sqrt(averageTestError);
        
        // Obtain training and test averages and standard deviations

        std::cout << "FINAL REPORT" << endl;
        std::cout << "************" << endl;
        std::cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        std::cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp(evalue, mvalue);

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            std::exit(-1);
        }

        // Reading training and test data: call to util::readData(...)
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            std::exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }

    
}

