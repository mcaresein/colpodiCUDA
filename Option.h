/*##########################################
# Tipi di Payoff utilizzabili implementati #
# nella libreria.                          #
##########################################*/

#ifndef _Option_h_
#define _Option_h_
#include "DataTypes.h"

class Option{
public:
    __device__ __host__ Option(OptionData);
    __device__ __host__ int GetNumberOfDatesToSimulate();
    __device__ __host__ double GetTInitial();
    __device__ __host__ double GetMaturityDate();
    __device__ __host__  virtual double GetPayOff(double*)=0;
protected:
    OptionData _OptionInput;
};

class OptionPlainVanillaCall: public Option{
public:
    __device__ __host__ OptionPlainVanillaCall(OptionData);
    __device__ __host__  double GetPayOff(double*);
};

class OptionPlainVanillaPut: public Option{
public:
    __device__ __host__ OptionPlainVanillaPut(OptionData);
    __device__ __host__  double GetPayOff(double*);
};

#endif
