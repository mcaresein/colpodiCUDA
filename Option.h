/*##########################################
# Tipi di Payoff utilizzabili implementati #
# nella libreria.                          #
##########################################*/

#ifndef _Option_h_
#define _Option_h_
#include "OptionData.h"
#include "UnderlyingPath.h"

class Option{
public:
    __device__ __host__ Option(OptionData, MontecarloPath*);
    __device__ __host__  virtual double GetPayOff()=0;
protected:
    OptionData _OptionParameters;
    MontecarloPath* _Path;
};

class OptionForward: public Option{
public:
    __device__ __host__ OptionForward(OptionData, MontecarloPath*);
    __device__ __host__  double GetPayOff();
};

class OptionPlainVanilla: public Option{
public:
    __device__ __host__ OptionPlainVanilla(OptionData, MontecarloPath*);
    __device__ __host__  double GetPayOff();
};

class OptionAbsolutePerformanceBarrier: public Option{
public:
    __device__ __host__ OptionAbsolutePerformanceBarrier(OptionData, MontecarloPath*);
    __device__ __host__  double GetPayOff();
};

#endif
