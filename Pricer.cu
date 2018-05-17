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

__device__ __host__ void MonteCarloPricer::GetPrice(){
    PayOffPlainVanilla pay;

    RandomGenerator* Generator=new CombinedGenerator(_S);

    StocasticProcess* Process=new EulerLogNormalProcess(_MarketInput.Volatility, _MarketInput.Drift);

    MontecarloPath Path(_MarketInput.SInitial, _OptionInput.TInitial, _OptionInput.TFinal, Generator, Process, _OptionInput.NSteps);

    for(int j=0; j<_NStreams; j++){
        float* value=new float[_OptionInput.NSteps];
        value=Path.GetPath();
        float payoff=pay.GetPayOff(value, _OptionInput.StrikePrice, _OptionInput.NSteps);
        _PayOff+=payoff;
        _PayOff2+=payoff*payoff;
        //delete[] value;
    }
    _PayOff=_PayOff;
    _PayOff2=_PayOff2;


};

__device__ __host__ float MonteCarloPricer::GetPayOff(){
    return _PayOff;
};

__device__ __host__ float MonteCarloPricer::GetPayOff2(){
    return _PayOff2;
};
