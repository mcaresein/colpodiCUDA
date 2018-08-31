#ifndef _Binomial_h_
#define _Binomial_h_
#include "Seed.h"
#include "RandomGenerator.h"

class Binomial: public RandomGenerator{
public:
    __host__ __device__  virtual double GetBinomialRandomNumber();

};

#endif
