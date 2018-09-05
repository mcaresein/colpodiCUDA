#ifndef _RandomGeneratorCombined_h_
#define _RandomGeneratorCombined_h_
#include "RandomGenerator.h"

class RandomGeneratorCombined: public RandomGenerator{
public:
    __host__ __device__  RandomGeneratorCombined(Seed);
    __host__ __device__  double GetUniformRandomNumber();
    //__host__ __device__  virtual void SetRandomNumber()=0;
protected:
    Seed _Seed;
private:
  __host__ __device__  unsigned int LCGStep(unsigned int &, unsigned int , unsigned long );
  __host__ __device__  unsigned int TausStep(unsigned int &, unsigned int , unsigned int , unsigned int , unsigned long );
};

#endif
