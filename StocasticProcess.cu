#include <iostream>
#include <cmath>
#include "RandomGenerator.h"
#include "StocasticProcess.h"

__host__ __device__  ExactLogNormalProcess::ExactLogNormalProcess(RandomGenerator* Generator){
  _Generator=Generator;
};

__host__ __device__  double ExactLogNormalProcess::Step(MarketData MarketInput, double TimeStep, double PriceStep){
    //double w=_Generator->GetGaussianRandomNumber();

    return PriceStep*exp((MarketInput.Drift - (MarketInput.Volatility*MarketInput.Volatility)/2)*TimeStep + MarketInput.Volatility*sqrt(TimeStep)*_Generator->GetGaussianRandomNumber());
};

__host__ __device__  EulerLogNormalProcess::EulerLogNormalProcess(RandomGenerator* Generator){
    _Generator=Generator;
};

__host__ __device__  double EulerLogNormalProcess::Step(MarketData MarketInput, double TimeStep, double PriceStep){
  return PriceStep*(1+MarketInput.Drift*TimeStep+MarketInput.Volatility*sqrt(TimeStep)*_Generator->GetGaussianRandomNumber());
};
