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


class MonteCarloPricer{
public:
    __device__ __host__ MonteCarloPricer(MarketData ,Option*, RandomGenerator*, StocasticProcess* , int);
    __device__ __host__ void ComputePrice();
    __device__ __host__ double GetPayOff();
    __device__ __host__ double GetPayOff2();
private:
    MarketData _MarketInput;
    Option* _Option;
    double _PayOff;
    double _PayOff2;
    int _NStreams;
    RandomGenerator* _Generator;
    StocasticProcess* _Process;
};

#endif
