/*#####################################################
# Classe che restituisce la somma e la somma quadrata #
# dei Pay Off calcolati in uno stream (ComputePrice). #
#####################################################*/

#ifndef _Pricer_h_
#define _Pricer_h_

#include <iostream>
#include "DataTypes.h"

class MonteCarloPricer{
public:
    __device__ __host__ MonteCarloPricer(MarketData ,OptionData , int, Seed);
    __device__ __host__ void ComputePrice();
    __device__ __host__ double GetPayOff();
    __device__ __host__ double GetPayOff2();
private:
    MarketData _MarketInput;
    OptionData _OptionInput;
    double _PayOff;
    double _PayOff2;
    int _NStreams;
    Seed _S;
};

#endif
