#include <iostream>
#include "DataTypes.h"
#ifndef _Pricer_h_
#define _Pricer_h_

class MonteCarloPricer{
public:
    __device__ __host__ MonteCarloPricer(MarketData ,OptionData , int, Seed);
    __device__ __host__ void GetPrice();
    __device__ __host__ float GetPayOff();
    __device__ __host__ float GetPayOff2();
private:
    MarketData _MarketInput;
    OptionData _OptionInput;
    float _PayOff;
    float _PayOff2;
    int _NStreams;
    Seed _S;
};

#endif
