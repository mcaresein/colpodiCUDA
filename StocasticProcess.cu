#include <iostream>
#include <cmath>
#include "RandomGenerator.h"
#include "StocasticProcess.h"

__host__ __device__  ExactLogNormalProcess::ExactLogNormalProcess(double Volatility, double Drift){
  _Volatility=Volatility;
  _Drift=Drift;
};

__host__ __device__  double ExactLogNormalProcess::Step(double InitialPrice, double TimeStep, double RandomNumber){
  return InitialPrice*exp((_Drift - (_Volatility*_Volatility)/2)*TimeStep + _Volatility*sqrt(TimeStep)*RandomNumber);
};

__host__ __device__  EulerLogNormalProcess::EulerLogNormalProcess(double Volatility, double Drift){
    _Volatility=Volatility;
    _Drift=Drift;
};

__host__ __device__  double EulerLogNormalProcess::Step(double InitialPrice, double TimeStep, double RandomNumber){
  return InitialPrice*(1+_Drift*TimeStep+_Volatility*sqrt(TimeStep)*RandomNumber);
};
