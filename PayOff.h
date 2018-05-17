#ifndef _PayOff_h_
#define _PayOff_h_

class PayOff{
public:
    __device__ __host__  virtual float GetPayOff(float*, float, int)=0;
};

class PayOffPlainVanilla: public PayOff{
public:
    __device__ __host__  float GetPayOff(float*, float, int);

};

#endif
