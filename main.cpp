/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <random>
#include <vector>
#include "Cell.h"
#include "Array.h"
#include "formula.h"
#include "NeuroSim.h"
#include "Param.h"
#include "IO.h"
#include "Train.h"
#include "Test.h"
#include "Mapping.h"
#include "Definition.h"
#include "omp.h"
#include "json.hpp"
using json = nlohmann::json;


int main(int argc, char ** argv) {
	gen.seed(0);

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <config file>" << std::endl;
		return 1;
	}

	std::ifstream config_stream(argv[1]);
	json config = json::parse(config_stream);
	config_stream.close();

	param->read_config(&config);
	//param->print();
	std::cout << "Config read from file: " << argv[1] << std::endl;
	
	/* Load in MNIST data */
	ReadTrainingDataFromFile("patch60000_train.txt", "label60000_train.txt");
	ReadTestingDataFromFile("patch10000_test.txt", "label10000_test.txt");

	std::string device_type;
	try
	{
		device_type = config.at("device-params").at("type");
	}
	catch(const std::exception& e)
	{
		std::cerr << "Device type not found in config ('device-params'->'type'): " << e.what() << std::endl;
		return 1;
	}

	if (device_type == "IdealDevice") {
		/* Initialization of synaptic array from input to hidden layer */
		arrayIH->Initialization<IdealDevice>(&config);
		/* Initialization of synaptic array from hidden to output layer */
		arrayHO->Initialization<IdealDevice>(&config);
	} else if (device_type == "RealDevice") {
		arrayIH->Initialization<RealDevice>(&config); 
		arrayHO->Initialization<RealDevice>(&config);
	} else if (device_type == "RealLogisticDevice") {
		arrayIH->Initialization<RealLogisticDevice>(&config);
		arrayHO->Initialization<RealLogisticDevice>(&config);
	} 
	else if (device_type == "MeasuredDevice") {
		arrayIH->Initialization<MeasuredDevice>(&config);
		arrayHO->Initialization<MeasuredDevice>(&config);
	} else if (device_type == "SRAM") {
		arrayIH->Initialization<SRAM>(&config, param->numWeightBit);
		arrayHO->Initialization<SRAM>(&config, param->numWeightBit);
	} else if (device_type == "DigitalNVM") {
		arrayIH->Initialization<DigitalNVM>(&config, param->numWeightBit,true);
		arrayHO->Initialization<DigitalNVM>(&config, param->numWeightBit,true);
	} else {
		std::cerr << "Invalid device type: " << device_type << std::endl;
		return 1;
	}

	//arrayIH->Initialization<HybridCell>(&config); // the 3T1C+2PCM cell (TODO)
	//arrayIH->Initialization<_2T1F>(&config); (TODO)

	//arrayHO->Initialization<HybridCell>(&config); // the 3T1C+2PCM cell (TODO)
	//arrayHO->Initialization<_2T1F>(&config); (TODO)

	std::cout << "Device type: " << device_type << std::endl;

    omp_set_num_threads(param->numThreads);
	/* Initialization of NeuroSim synaptic cores */
	param->relaxArrayCellWidth = 0;
	NeuroSimSubArrayInitialize(subArrayIH, arrayIH, inputParameterIH, techIH, cellIH);
	param->relaxArrayCellWidth = 1;
	NeuroSimSubArrayInitialize(subArrayHO, arrayHO, inputParameterHO, techHO, cellHO);
	/* Calculate synaptic core area */
	NeuroSimSubArrayArea(subArrayIH);
	NeuroSimSubArrayArea(subArrayHO);
	
	/* Calculate synaptic core standby leakage power */
	NeuroSimSubArrayLeakagePower(subArrayIH);
	NeuroSimSubArrayLeakagePower(subArrayHO);
	
	/* Initialize the neuron peripheries */
	NeuroSimNeuronInitialize(subArrayIH, inputParameterIH, techIH, cellIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH);
	NeuroSimNeuronInitialize(subArrayHO, inputParameterHO, techHO, cellHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO);
	/* Calculate the area and standby leakage power of neuron peripheries below subArrayIH */
	double heightNeuronIH, widthNeuronIH;
	NeuroSimNeuronArea(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH, &heightNeuronIH, &widthNeuronIH);
	double leakageNeuronIH = NeuroSimNeuronLeakagePower(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH);
	/* Calculate the area and standby leakage power of neuron peripheries below subArrayHO */
	double heightNeuronHO, widthNeuronHO;
	NeuroSimNeuronArea(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO, &heightNeuronHO, &widthNeuronHO);
	double leakageNeuronHO = NeuroSimNeuronLeakagePower(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO);
	
	/* Print the area of synaptic core and neuron peripheries */
	double totalSubArrayArea = subArrayIH->usedArea + subArrayHO->usedArea;
	double totalNeuronAreaIH = adderIH.area + muxIH.area + muxDecoderIH.area + dffIH.area + subtractorIH.area;
	double totalNeuronAreaHO = adderHO.area + muxHO.area + muxDecoderHO.area + dffHO.area + subtractorHO.area;
	printf("Total SubArray (synaptic core) area=%.4e m^2\n", totalSubArrayArea);
	printf("Total Neuron (neuron peripheries) area=%.4e m^2\n", totalNeuronAreaIH + totalNeuronAreaHO);
	printf("Total area=%.4e m^2\n", totalSubArrayArea + totalNeuronAreaIH + totalNeuronAreaHO);

	/* Print the standby leakage power of synaptic core and neuron peripheries */
	printf("Leakage power of subArrayIH is : %.4e W\n", subArrayIH->leakage);
	printf("Leakage power of subArrayHO is : %.4e W\n", subArrayHO->leakage);
	printf("Leakage power of NeuronIH is : %.4e W\n", leakageNeuronIH);
	printf("Leakage power of NeuronHO is : %.4e W\n", leakageNeuronHO);
	printf("Total leakage power of subArray is : %.4e W\n", subArrayIH->leakage + subArrayHO->leakage);
	printf("Total leakage power of Neuron is : %.4e W\n", leakageNeuronIH + leakageNeuronHO);
	
	/* Initialize weights and map weights to conductances for hardware implementation */
	WeightInitialize();
	if (param->useHardwareInTraining)
    	WeightToConductance();
	srand(0);	// Pseudorandom number seed
	
	std::ofstream mywriteoutfile;
	mywriteoutfile.open("output.csv");
	int totalArraySize = param->nInput*param->nHide + param->nHide*param->nOutput;                                                                                                        
	for (int i=1; i<=param->totalNumEpochs/param->interNumEpochs; i++){
		Train(param->numTrainImagesPerEpoch, param->interNumEpochs,param->optimization_type);
		if (!param->useHardwareInTraining && param->useHardwareInTestingFF) { WeightToConductance(); }
		Validate();
        if (HybridCell *temp = dynamic_cast<HybridCell*>(arrayIH->cell[0][0]))
            WeightTransfer();
        else if(_2T1F *temp = dynamic_cast<_2T1F*>(arrayIH->cell[0][0]))
            WeightTransfer_2T1F();
                
		mywriteoutfile << i*param->interNumEpochs << ", " << (double)correct/param->numMnistTestImages*100 << std::endl;
		
		printf("Accuracy at %d epochs is : %.2f%\n", i*param->interNumEpochs, (double)correct/param->numMnistTestImages*100);
		/* Here the performance metrics of subArray also includes that of neuron peripheries (see Train.cpp and Test.cpp) */
		printf("\tRead latency=%.4e s\n", subArrayIH->readLatency + subArrayHO->readLatency);
		printf("\tWrite latency=%.4e s\n", subArrayIH->writeLatency + subArrayHO->writeLatency);
		printf("\tRead energy=%.4e J\n", arrayIH->readEnergy + subArrayIH->readDynamicEnergy + arrayHO->readEnergy + subArrayHO->readDynamicEnergy);
		printf("\tWrite energy=%.4e J\n", arrayIH->writeEnergy + subArrayIH->writeDynamicEnergy + arrayHO->writeEnergy + subArrayHO->writeDynamicEnergy);
		if(HybridCell* temp = dynamic_cast<HybridCell*>(arrayIH->cell[0][0])){
            printf("\tTransfer latency=%.4e s\n", subArrayIH->transferLatency + subArrayHO->transferLatency);
            printf("\tTransfer latency=%.4e s\n", subArrayIH->transferLatency);	
            printf("\tTransfer energy=%.4e J\n", arrayIH->transferEnergy + subArrayIH->transferDynamicEnergy + arrayHO->transferEnergy + subArrayHO->transferDynamicEnergy);
        }
        else if(_2T1F* temp = dynamic_cast<_2T1F*>(arrayIH->cell[0][0])){
            printf("\tTransfer latency=%.4e s\n", subArrayIH->transferLatency);	
            printf("\tTransfer energy=%.4e J\n", arrayIH->transferEnergy + subArrayIH->transferDynamicEnergy + arrayHO->transferEnergy + subArrayHO->transferDynamicEnergy);
         }
        printf("\tThe total weight update=%.4e\n", totalWeightUpdate);
        printf("\tThe total pulse number=%.4e\n", totalNumPulse);
		printf("\tThe total actual conductance update=%.4e\n", actualConductanceUpdate);
		printf("\tThe total actual pulse number=%.4e\n", actualNumPulse);
		printf("\tThe total actual conductance update per synapse=%.4e\n", actualConductanceUpdate/totalArraySize);
		printf("\tThe total actual pulse number per synapse=%.4e\n", actualNumPulse/totalArraySize);
	}
	// print the summary: 
	printf("\n");
	return 0;
}


