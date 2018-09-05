#ifndef _RandomGeneratorCombinedBinomial_h_
#define _RandomGeneratorCombinedBinomial_h_

#include  "RandomGeneratorCombined.h"
#include  "Binomial.h"

class RandomGeneratorCombinedBinomial: public RandomGeneratorCombined, public Binomial{
public:
  __host__ __device__ RandomGeneratorCombinedBinomial(Seed seed): RandomGeneratorCombined(seed), Binomial(){};
  __host__ __device__ void SetRandomNumber();
};

#endif
