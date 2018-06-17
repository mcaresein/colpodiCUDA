#include "MonteCarloPricer.h"
#include "UnderlyingPath.h"
#include "StocasticProcess.h"
#include "Option.h"
#include "Seed.h"
#include "MarketData.h"
#include "OptionData.h"
#include "GPUData.h"
#include "SimulationParameters.h"


__device__ __host__ MonteCarloPricer::MonteCarloPricer(Option* Option, StocasticProcess* Process, int NStreams){
//    _MarketInput=MarketInput;
    _NStreams=NStreams;
    _Option=Option;
    _Process=Process;
};

//## Metodo per il calcolo delle somme semplici e quadrate dei PayOff simulati in uno stream. ##

__device__ __host__ void MonteCarloPricer::ComputePrice(Statistics* PayOffs){

    for(int j=0; j<_NStreams; j++){
        double* value=new double[_Option->GetNumberOfDatesToSimulate()];
        value=_Option->GetMontecarloPath()->GetPath();
        double payoff=_Option->GetPayOff(value);
        PayOffs->AddValue(payoff);
    }
};
