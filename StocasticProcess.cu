#include <iostream>
#include <cmath>
#include "RandomGenerator.h"
#include "StocasticProcess.h"

__host__ __device__  ExactLogNormalProcess::ExactLogNormalProcess(double volatility, double drift){
  _volatility=volatility;
  _drift=drift;
};

__host__ __device__  double ExactLogNormalProcess::Step(double S, double T, double w){
  return S*exp((_drift - (_volatility*_volatility)/2)*T + _volatility*sqrt(T)*w);
};

__host__ __device__  EulerLogNormalProcess::EulerLogNormalProcess(double volatility, double drift){
  _volatility=volatility;
  _drift=drift;
};

__host__ __device__  double EulerLogNormalProcess::Step(double S, double T, double w){
  return S*(1+_drift*T+_volatility*sqrt(T)*w);
};
