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
    bool tflag = 0, Tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0, wflag = 0, pflag = 0;    //Changed by me
    char *tvalue = NULL, *Tvalue = NULL, *wvalue = NULL;    //Changed by me
    int c, ivalue, lvalue, hvalue;
    float evalue = 0.0, mvalue = 0.0;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:w:p")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 't':   //Added by my
                tflag = true;
                tvalue = optarg;
                break;
            case 'T':   //Added by my
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'i':   //Added by me
                iflag = true;   //Used to change the number of iteration if is true or default if is false
                /* SHOULD I CHECK IF INPUT IS A NUMBER? */
                ivalue = std::stoi(optarg);
                break;
            case 'l':   //Added by me
                lflag = true;   //Used to change the number of hidden layers if is true or default if is false
                /* SHOULD I CHECK IF INPUT IS A NUMBER? */
                lvalue = std::stoi(optarg);
                break;
            case 'h':   //Added by me
                hflag = true;   //Used to change the number of neurons of hidden layers if is true or default if is false
                /* SHOULD I CHECK IF INPUT IS A NUMBER? */
                hvalue = std::stoi(optarg);
                break;
            case 'e':   //Added by me
                eflag = true;   //Used to change eta parameter if is true or default if is false
                /* SHOULD I CHECK IF INPUT IS A NUMBER? */
                evalue = std::stof(optarg);
                break;
            case 'm':   //Added by me
                mflag = true;   //Used to change mu parameter if is true or default if is false
                /* SHOULD I CHECK IF INPUT IS A NUMBER? */
                mvalue = std::stof(optarg);
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case '?':
                if (optopt == 't' || optopt == 'T' || optopt == 'w' || optopt == 'p')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }


    //Added by me
    //Check if there are missing arguments and set some argument to default
    ///////////////////////////////////////////////////////////////////

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value;
    	int iterations = ivalue; // This should be corrected    //CHanged by me

        // Read training and test data: call to util::readData(...)
    	Dataset * trainDataset = util::readData(tvalue); // This should be corrected    //Changed by me
    	Dataset * testDataset = util::readData(Tvalue); // This should be corrected     //Changed by me

        // Initialize topology vector
    	int layers= lvalue; // This should be corrected //Changed by me
    	std::vector<int> topology= std::vector<int>(lvalue+2, hvalue); // This should be corrected //Changed by me
        //Added by me //Last layer (output layer)
        topology[lvalue+1] = 1;

        // Initialize the network using the topology vector
        mlp.initialize(layers+2,topology);


        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
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
        MultilayerPerceptron mlp;

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

