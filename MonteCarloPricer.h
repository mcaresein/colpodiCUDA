/*#####################################################
# Classe che restituisce la somma e la somma quadrata #
# dei Pay Off calcolati in uno stream (ComputePrice). #
#####################################################*/

#ifndef _MonteCarloPricer_h_
#define _MonteCarloPricer_h_

#include <iostream>
#include "RandomGenerator.h"
#include "DataTypes.h"
#include "Option.h"
#include "StocasticProcess.h"
#include "Statistics.h"

class MonteCarloPricer{
public:
    __device__ __host__ MonteCarloPricer(MarketData ,Option*, RandomGenerator*, StocasticProcess* , int);
    __device__ __host__ void ComputePrice(DevStatistics*);
private:
    MarketData _MarketInput;
    Option* _Option;
    int _NStreams;
    RandomGenerator* _Generator;
    StocasticProcess* _Process;
};

#endif
