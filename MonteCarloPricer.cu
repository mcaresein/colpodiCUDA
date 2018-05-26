#include "MonteCarloPricer.h"
#include "RandomGenerator.h"
#include "UnderlyingPath.h"
#include "StocasticProcess.h"
#include "Option.h"
#include "DataTypes.h"


__device__ __host__ MonteCarloPricer::MonteCarloPricer(MarketData MarketInput, Option* Option , RandomGenerator* Generator, StocasticProcess* Process, int NStreams){
    _MarketInput=MarketInput;
    _PayOff=0;
    _PayOff2=0;
    _NStreams=NStreams;
    _Generator=Generator;
    _Option=Option;
    _Process=Process;
};

//## Metodo per il calcolo delle somme semplici e quadrate dei PayOff simulati in uno stream. ##

__device__ __host__ void MonteCarloPricer::ComputePrice(){

    MontecarloPath Path(_MarketInput.EquityInitialPrice, _Option , _Generator, _Process);

    for(int j=0; j<_NStreams; j++){
        double* value=new double[_Option->GetNumberOfDatesToSimulate()];
        value=Path.GetPath();
        double payoff=_Option->GetPayOff(value);
        _PayOff+=payoff;
        _PayOff2+=payoff*payoff;
    }
    _PayOff=_PayOff;
    _PayOff2=_PayOff2;

};

__device__ __host__ double MonteCarloPricer::GetPayOff(){
    return _PayOff;
};

__device__ __host__ double MonteCarloPricer::GetPayOff2(){
    return _PayOff2;
};
