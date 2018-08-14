#ifndef _RandomGeneratorCombined_h_
#define _RandomGeneratorCombined_h_

#include "Seed.h"
#include "RandomGenerator.h"

class RandomGeneratorCombined: public RandomGenerator{
public:
    __host__ __device__  RandomGeneratorCombined(Seed, bool);
    __host__ __device__  double GetUniformRandomNumber();
private:
    __host__ __device__  unsigned int LCGStep(unsigned int &, unsigned int , unsigned long );
    __host__ __device__  unsigned int TausStep(unsigned int &, unsigned int , unsigned int , unsigned int , unsigned long );
    Seed _Seed;
};

#endif
