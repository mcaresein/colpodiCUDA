#ifndef _MainFunctions_cu_
#define _MainFunctions_cu_

#include <iostream>
#include "MonteCarloPricer.h"
#include "Statistics.h"
#include "RandomGenerator.h"
#include "RandomGeneratorCombined.h"
#include "Seed.h"
#include "MarketData.h"
#include "OptionData.h"
#include "SimulationParameters.h"
#include "Option.h"

/*############################ Kernel Functions ############################*/

__host__ __device__ void TrueKernel(Seed* SeedVector, Statistics* PayOffs, int streams, MarketData MarketInput, OptionData OptionInput, SimulationParameters Parameters, int cont){

    RandomGenerator* Generator= new RandomGeneratorCombined(SeedVector[cont], false);

    StocasticProcess* Process;
    if(Parameters.EulerApprox==false)
        Process=new ExactLogNormalProcess(Generator);
    if(Parameters.EulerApprox==true)
        Process=new EulerLogNormalProcess(Generator);

    MontecarloPath* Path=new MontecarloPath(MarketInput, OptionInput.MaturityDate, OptionInput.NumberOfDatesToSimulate, Process, OptionInput.EulerSubStep);

    Option* Option;
    if(Parameters.OptionType==0)
        Option=new OptionForward(OptionInput, Path);
    if(Parameters.OptionType==1 || Parameters.OptionType==2)
        Option=new OptionPlainVanilla(OptionInput, Path);
    if(Parameters.OptionType==3)
        Option=new OptionAbsolutePerformanceBarrier(OptionInput, Path, MarketInput.Volatility);

    MonteCarloPricer Pricer(Option, Process, streams);

    PayOffs[cont].Reset();
    Pricer.ComputePrice(&PayOffs[cont]);

}

__global__ void Kernel(Seed* SeedVector, Statistics* PayOffs, int streams, MarketData MarketInput, OptionData OptionInput, SimulationParameters Parameters){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    TrueKernel(SeedVector,PayOffs, streams, MarketInput, OptionInput, Parameters, i);
};

//## Funzione che gira su CPU che restituisce due vettori con sommme dei PayOff e dei PayOff quadrati. ##

__host__ void KernelSimulator(Seed* SeedVector, Statistics* PayOffs, int streams, MarketData MarketInput, OptionData OptionInput, SimulationParameters Parameters, int threads){

    for(int i=0; i<threads; i++) TrueKernel(SeedVector, PayOffs, streams, MarketInput, OptionInput, Parameters, i);

};

#endif
