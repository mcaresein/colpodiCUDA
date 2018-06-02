#ifndef _MainFunctions_cu_
#define _MainFunctions_cu_

#include <iostream>
#include "MonteCarloPricer.h"
#include "Statistics.h"
#include "RandomGenerator.h"
#include "DataTypes.h"
#include "Option.h"

/*############################ Kernel Functions ############################*/

__host__ __device__ void TrueKernel(Seed* SeedVector, DevStatistics* PayOffs, int streams, MarketData MarketInput, OptionData OptionInput, int cont){

    RandomGenerator* Generator= new CombinedGenerator(SeedVector[cont], true);
    Option* Option=new OptionPlainVanillaCall(OptionInput);
    StocasticProcess* Process=new ExactLogNormalProcess(MarketInput.Volatility, MarketInput.Drift);

    MonteCarloPricer Pricer(MarketInput, Option, Generator, Process, streams);

//    DevStatistics* temp=new DevStatistics;  non va
    DevStatistics temp;
    Pricer.ComputePrice(&temp);
//  Pricer.ComputePrice(&PayOffs[cont]);  perchè non va???
    PayOffs[cont]=temp;
}

__global__ void Kernel(Seed* SeedVector, DevStatistics* PayOffs, int streams, MarketData MarketInput, OptionData OptionInput){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    TrueKernel(SeedVector,PayOffs, streams, MarketInput, OptionInput, i);
};

//## Funzione che gira su CPU che restituisce due vettori con sommme dei PayOff e dei PayOff quadrati. ##

__host__ void KernelSimulator(Seed* SeedVector, DevStatistics* PayOffs, int streams, MarketData MarketInput, OptionData OptionInput, int threads){

    for(int i=0; i<threads; i++) TrueKernel(SeedVector, PayOffs, streams, MarketInput, OptionInput, i);

};

/*
__global__ void Kernel(Seed* SeedVector, DevStatistics* PayOffs, int streams, MarketData MarketInput, OptionData OptionInput){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    RandomGenerator* Generator= new CombinedGenerator(SeedVector[i], true);
    Option* Option=new OptionPlainVanillaCall(OptionInput);
    StocasticProcess* Process=new ExactLogNormalProcess(MarketInput.Volatility, MarketInput.Drift);

    MonteCarloPricer Pricer(MarketInput, Option, Generator, Process, streams);
    DevStatistics temp;
    Pricer.ComputePrice(&temp);
    PayOffs[i]=temp;
};
*/

#endif
