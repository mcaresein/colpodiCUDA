#include <iostream>
#include <cmath>
#include "RandomGenerator.h"
#include "StocasticProcess.h"

__host__ __device__  ExactLogNormalProcess::ExactLogNormalProcess(float volatility, float drift){
  _volatility=volatility;
  _drift=drift;
};

__host__ __device__  float ExactLogNormalProcess::Step(float S, float T, float w){
  return S*exp((_drift - pow(_volatility,2)/2)*T + _volatility*sqrt(T)*w);
};

__host__ __device__  EulerLogNormalProcess::EulerLogNormalProcess(float volatility, float drift){
  _volatility=volatility;
  _drift=drift;
};

__host__ __device__  float EulerLogNormalProcess::Step(float S, float T, float w){
  return S*(1+_drift*T+_volatility*sqrt(T)*w);
};
