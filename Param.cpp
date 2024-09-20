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

#include <string>
#include <fstream>
#include <iostream>
#include "math.h"
#include "Param.h"
#include "json.hpp"
using json = nlohmann::json;

Param::Param() { 
}

void Param::init(std::string filename) {
	std::ifstream f(filename.c_str());
	json data = json::parse(f);
	f.close();

	json p = data.at("simulation-params");

	/* MNIST dataset */
	numMnistTrainImages = p.at("mnist").value("numMnistTrainImages", 60000);// # of training images in MNIST
	numMnistTestImages = p.at("mnist").value("numMnistTestImages", 10000);	// # of testing images in MNIST
	
	/* Algorithm parameters */
	numTrainImagesPerEpoch = p.at("algorithm").value("numTrainImagesPerEpoch", 8000);	// # of training images per epoch 
    numTrainImagesPerBatch = p.at("algorithm").value("numTrainImagesPerBatch", 1);   // # of training images per batch. It is 1 for SGD
	totalNumEpochs = p.at("algorithm").value("totalNumEpochs", 125);	// Total number of epochs
	interNumEpochs = p.at("algorithm").value("interNumEpochs", 1);		// Internal number of epochs (print out the results every interNumEpochs)
	nInput = p.at("algorithm").value("nInput", 400);     // # of neurons in input layer
	nHide = p.at("algorithm").value("nHide", 100);      // # of neurons in hidden layer
	nOutput = p.at("algorithm").value("nOutput", 10);     // # of neurons in output layer
	alpha1 = p.at("algorithm").value("alpha1", 0.4);	// Learning rate for the weights from input to hidden layer
	alpha2 = p.at("algorithm").value("alpha2", 0.2);	// Learning rate for the weights from hidden to output layer
	maxWeight = p.at("algorithm").value("maxWeight", 1);	// Upper bound of weight value
	minWeight = p.at("algorithm").value("minWeight", -1);	// Lower bound of weight value
	/*Optimization method 
	Available option include: "SGD", "Momentum", "RMSprop" and "Adam"*/
	optimization_type = (char*) p.at("algorithm").value("optimization_type", "SGD").c_str();


	/* Hardware parameters */
	useHardwareInTrainingFF = p.at("hardware").value("useHardwareInTrainingFF", true);   // Use hardware in the feed forward part of training or not (true: realistic hardware, false: ideal software)
	useHardwareInTrainingWU = p.at("hardware").value("useHardwareInTrainingWU", true);   // Use hardware in the weight update part of training or not (true: realistic hardware, false: ideal software)
	useHardwareInTraining = useHardwareInTrainingFF || useHardwareInTrainingWU;    // Use hardware in the training or not
	useHardwareInTestingFF = p.at("hardware").value("useHardwareInTestingFF", true);    // Use hardware in the feed forward part of testing or not (true: realistic hardware, false: ideal software)
	numBitInput = p.at("hardware").value("numBitInput", 1);       // # of bits of the input data (=1 for black and white data)
	numBitPartialSum = p.at("hardware").value("numBitPartialSum", 8);  // # of bits of the digital output (partial weighted sum output)
	pSumMaxHardware = pow(2, numBitPartialSum) - 1;   // Max digital output value of partial weighted sum
	numInputLevel = pow(2, numBitInput);  // # of levels of the input data
	numWeightBit = p.at("hardware").value("numWeightBit", 6);	// # of weight bits (only for pure algorithm, SRAM and digital RRAM hardware)
	BWthreshold = p.at("hardware").value("BWthreshold", 0.5);	// The black and white threshold for numBitInput=1
	Hthreshold = p.at("hardware").value("Hthreshold", 0.5);	// The spiking threshold for the hidden layer (da1 in Train.cpp and Test.cpp)
	numColMuxed = p.at("hardware").value("numColMuxed", 16);	// How many columns share 1 read circuit (for analog RRAM) or 1 S/A (for digital RRAM)
	numWriteColMuxed = p.at("hardware").value("numWriteColMuxed", 16);	// How many columns share 1 write column decoder driver (for digital RRAM)
	writeEnergyReport = p.at("hardware").value("writeEnergyReport", true);	// Report write energy calculation or not
	NeuroSimDynamicPerformance = p.at("hardware").value("NeuroSimDynamicPerformance", true); // Report the dynamic performance (latency and energy) in NeuroSim or not
	relaxArrayCellHeight = p.at("hardware").value("relaxArrayCellHeight", 0);	// True: relax the array cell height to standard logic cell height in the synaptic array
	relaxArrayCellWidth = p.at("hardware").value("relaxArrayCellWidth", 0);	// True: relax the array cell width to standard logic cell width in the synaptic array
	arrayWireWidth = p.at("hardware").value("arrayWireWidth", 100);	// Array wire width (nm)
	processNode = p.at("hardware").value("processNode", 32);	// Technology node (nm)
	clkFreq = p.at("hardware").value("clkFreq", 2e9);		// Clock frequency (Hz)

}

void Param::print() {
	std::cout << "numMnistTrainImages: " << numMnistTrainImages << std::endl;
	std::cout << "numMnistTestImages: " << numMnistTestImages << std::endl;
	std::cout << "numTrainImagesPerEpoch: " << numTrainImagesPerEpoch << std::endl;
	std::cout << "numTrainImagesPerBatch: " << numTrainImagesPerBatch << std::endl;
	std::cout << "totalNumEpochs: " << totalNumEpochs << std::endl;
	std::cout << "interNumEpochs: " << interNumEpochs << std::endl;
	std::cout << "nInput: " << nInput << std::endl;
	std::cout << "nHide: " << nHide << std::endl;
	std::cout << "nOutput: " << nOutput << std::endl;
	std::cout << "alpha1: " << alpha1 << std::endl;
	std::cout << "alpha2: " << alpha2 << std::endl;
	std::cout << "maxWeight: " << maxWeight << std::endl;
	std::cout << "minWeight: " << minWeight << std::endl;
	std::cout << "optimization_type: " << optimization_type << std::endl;
	std::cout << "useHardwareInTrainingFF: " << useHardwareInTrainingFF << std::endl;
	std::cout << "useHardwareInTrainingWU: " << useHardwareInTrainingWU << std::endl;
	std::cout << "useHardwareInTraining: " << useHardwareInTraining << std::endl;
	std::cout << "useHardwareInTestingFF: " << useHardwareInTestingFF << std::endl;
	std::cout << "numBitInput: " << numBitInput << std::endl;
	std::cout << "numBitPartialSum: " << numBitPartialSum << std::endl;
	std::cout << "pSumMaxHardware: " << pSumMaxHardware << std::endl;
	std::cout << "numInputLevel: " << numInputLevel << std::endl;
	std::cout << "numWeightBit: " << numWeightBit << std::endl;
	std::cout << "BWthreshold: " << BWthreshold << std::endl;
	std::cout << "Hthreshold: " << Hthreshold << std::endl;
	std::cout << "numColMuxed: " << numColMuxed << std::endl;
	std::cout << "numWriteColMuxed: " << numWriteColMuxed << std::endl;
	std::cout << "writeEnergyReport: " << writeEnergyReport << std::endl;
	std::cout << "NeuroSimDynamicPerformance: " << NeuroSimDynamicPerformance << std::endl;
	std::cout << "relaxArrayCellHeight: " << relaxArrayCellHeight << std::endl;
	std::cout << "relaxArrayCellWidth: " << relaxArrayCellWidth << std::endl;
	std::cout << "arrayWireWidth: " << arrayWireWidth << std::endl;
	std::cout << "processNode: " << processNode << std::endl;
	std::cout << "clkFreq: " << clkFreq << std::endl;
}

