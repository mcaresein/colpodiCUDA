#include "Pricer.h"
#include "RandomGenerator.h"
#include "UnderlyingPath.h"
#include "StocasticProcess.h"
#include "PayOff.h"
#include "DataTypes.h"


__device__ __host__ MonteCarloPricer::MonteCarloPricer(MarketData MarketInput, OptionData OptionInput, int NStreams, Seed S){
    _MarketInput=MarketInput;
    _OptionInput=OptionInput;
    _PayOff=0;
    _PayOff2=0;
    _NStreams=NStreams;
    _S=S;
};

//## Metodo per il calcolo delle somme semplici e quadrate dei PayOff simulati in uno stream. ##

__device__ __host__ void MonteCarloPricer::ComputePrice(){

    PayOffPlainVanillaCall pay;
    RandomGenerator* Generator=new CombinedGenerator(_S);
    StocasticProcess* Process=new EulerLogNormalProcess(_MarketInput.Volatility, _MarketInput.Drift);

    MontecarloPath Path(_MarketInput.SInitial, _OptionInput.TInitial, _OptionInput.TFinal, Generator, Process, _OptionInput.NSteps);

    for(int j=0; j<_NStreams; j++){
        double* value=new double[_OptionInput.NSteps];
        value=Path.GetPath();
        double payoff=pay.GetPayOff(value, _OptionInput.StrikePrice, _OptionInput.NSteps);
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
