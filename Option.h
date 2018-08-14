/*##########################################
# Tipi di Payoff utilizzabili implementati #
# nella libreria.                          #
##########################################*/

#ifndef _Option_h_
#define _Option_h_
#include "OptionData.h"
#include "DatesVector.h"

class Option{
public:
    __device__ __host__ Option(OptionData);
    __device__ __host__  virtual double GetPayOff(DatesVector)=0;
protected:
    OptionData _OptionParameters;
};

class OptionForward: public Option{
public:
    __device__ __host__ OptionForward(OptionData);
    __device__ __host__  double GetPayOff(DatesVector);
};

class OptionPlainVanilla: public Option{
public:
    __device__ __host__ OptionPlainVanilla(OptionData);
    __device__ __host__  double GetPayOff(DatesVector);
};

class OptionAbsolutePerformanceBarrier: public Option{
public:
    __device__ __host__ OptionAbsolutePerformanceBarrier(OptionData, double, double);
    __device__ __host__  double GetPayOff(DatesVector);
private:
    double _EquityInitialPrice;
    double _Volatility;
};

#endif
