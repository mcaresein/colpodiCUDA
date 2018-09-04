#ifndef _Binomial_h_
#define _Binomial_h_

#include "RandomGenerator.h"

class Binomial {
public:
    __host__ __device__  double GetBinomialRandomNumber(RandomGenerator* Generator);
};

#endif
