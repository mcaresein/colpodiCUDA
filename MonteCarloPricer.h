/*#####################################################
# Classe che restituisce la somma e la somma quadrata #
# dei Pay Off calcolati in uno stream (ComputePrice). #
#####################################################*/

#ifndef _MonteCarloPricer_h_
#define _MonteCarloPricer_h_

#include <iostream>
#include "Seed.h"
#include "MarketData.h"
#include "OptionData.h"
#include "GPUData.h"
#include "SimulationParameters.h"
#include "Option.h"
#include "StocasticProcess.h"
#include "Statistics.h"

class MonteCarloPricer{
public:
    __device__ __host__ MonteCarloPricer(Option*, StocasticProcess* , int);
    __device__ __host__ void ComputePrice(Statistics*);
private:
    MarketData _MarketInput;
    Option* _Option;
    int _NStreams;
    StocasticProcess* _Process;
};

#endif
