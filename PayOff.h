/*##########################################
# Tipi di Payoff utilizzabili implementati #
# nella libreria.                          #
##########################################*/

#ifndef _PayOff_h_
#define _PayOff_h_

class PayOff{
public:
    __device__ __host__  virtual double GetPayOff(double*, double, int)=0;
};

class PayOffPlainVanillaCall: public PayOff{
public:
    __device__ __host__  double GetPayOff(double*, double, int);

};

class PayOffPlainVanillaPut: public PayOff{
public:
    __device__ __host__  double GetPayOff(double*, double, int);

};

#endif
