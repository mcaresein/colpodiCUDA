/*##########################################
# Tipi di Payoff utilizzabili implementati #
# nella libreria.                          #
##########################################*/

#ifndef _Option_h_
#define _Option_h_
#include "Seed.h"
#include "MarketData.h"
#include "OptionData.h"
#include "GPUData.h"
#include "SimulationParameters.h"
#include "UnderlyingPath.h"

class Option{
public:
    __device__ __host__ Option(OptionData, MontecarloPath*);
    __device__ __host__ int GetNumberOfDatesToSimulate();
//    __device__ __host__ int GetEulerSubStep();
//    __device__ __host__ double GetMaturityDate();
    __device__ __host__  virtual double GetPayOff(double*)=0;
    __device__ __host__ MontecarloPath* GetMontecarloPath();
protected:
    OptionData _OptionInput;
    MontecarloPath* _Path;
};

class OptionForward: public Option{
public:
    __device__ __host__ OptionForward(OptionData, MontecarloPath*);
    __device__ __host__  double GetPayOff(double*);
};

class OptionPlainVanilla: public Option{
public:
    __device__ __host__ OptionPlainVanilla(OptionData, MontecarloPath*);
    __device__ __host__  double GetPayOff(double*);
};

class OptionAbsolutePerformanceBarrier: public Option{
public:
    __device__ __host__ OptionAbsolutePerformanceBarrier(OptionData, MontecarloPath*, double Volatility, double InitialPrice);
    __device__ __host__  double GetPayOff(double*);
private:
    double _Volatility;
    double _InitialPrice;
};

#endif
