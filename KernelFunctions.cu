#ifndef _KernelFunctions_cu_
#define _KernelFunctions_cu_

#define RE_EXTRACTION_BOX_MULLER false

#include "MonteCarloPricer.h"
#include "Statistics.h"
#include "RandomGenerator.h"
#include "RandomGeneratorCombined.h"
#include "Seed.h"
#include "MarketData.h"
#include "OptionData.h"
#include "SimulationParameters.h"
#include "Option.h"
#include "UnderlyingAnagraphy.h"
#include "UnderlyingPrice.h"

/*############################ Kernel Functions ############################*/

__host__ __device__ void TrueKernel(Seed* SeedVector, Statistics* PayOffs, int streams, MarketData MarketInput, OptionDataContainer OptionInput, SimulationParameters Parameters, int cont){

    RandomGenerator* Generator= new RandomGeneratorCombined(SeedVector[cont], RE_EXTRACTION_BOX_MULLER);

    StocasticProcess* Process;
    if(Parameters.EulerApprox==false)
        Process=new ExactLogNormalProcess(Generator);
    if(Parameters.EulerApprox==true)
        Process=new EulerLogNormalProcess(Generator);

    UnderlyingAnagraphy* Anagraphy=new UnderlyingAnagraphy(MarketInput);
    UnderlyingPrice* Price=new UnderlyingPrice(Anagraphy);

    MonteCarloPath* Path=new MonteCarloPath(Price, MarketInput.EquityInitialPrice, OptionInput.MaturityDate, OptionInput.NumberOfFixingDate, Parameters.EulerSubStep);

    OptionData OptionParameters;
    OptionParameters.MaturityDate=OptionInput.MaturityDate;
    OptionParameters.NumberOfFixingDate=OptionInput.NumberOfFixingDate;
    OptionParameters.OptionType=OptionInput.OptionType;

    Option* Option;
    if( OptionInput.OptionType==0){
        Option=new OptionForward(OptionParameters);
    }
    if( OptionInput.OptionType==1 || OptionInput.OptionType==2){
        OptionParameters.AdditionalParameters=new double[1];
        OptionParameters.AdditionalParameters[0]=OptionInput.StrikePrice;

        Option=new OptionPlainVanilla(OptionParameters);
    }

    if( OptionInput.OptionType==3){
        OptionParameters.AdditionalParameters=new double[3];
        OptionParameters.AdditionalParameters[0]=OptionInput.B;
        OptionParameters.AdditionalParameters[1]=OptionInput.K;
        OptionParameters.AdditionalParameters[2]=OptionInput.N;

        Option=new OptionAbsolutePerformanceBarrier(OptionParameters, MarketInput.Volatility, MarketInput.EquityInitialPrice);
    }

    MonteCarloPricer Pricer(Option, Path, Process, streams, Parameters.AntitheticVariable);

    PayOffs[cont].Reset();
    Pricer.ComputePrice(&PayOffs[cont]);

}

__global__ void Kernel(Seed* SeedVector, Statistics* PayOffs, int streams, MarketData MarketInput, OptionDataContainer OptionInput, SimulationParameters Parameters){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    TrueKernel(SeedVector,PayOffs, streams, MarketInput, OptionInput, Parameters, i);
};

//## Funzione che gira su CPU che restituisce due vettori con sommme dei PayOff e dei PayOff quadrati. ##

__host__ void KernelSimulator(Seed* SeedVector, Statistics* PayOffs, int streams, MarketData MarketInput, OptionDataContainer OptionInput, SimulationParameters Parameters, int threads){

    for(int i=0; i<threads; i++) TrueKernel(SeedVector, PayOffs, streams, MarketInput, OptionInput, Parameters, i);

};

#endif
