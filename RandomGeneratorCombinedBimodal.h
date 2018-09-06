#ifndef _RandomGeneratorCombinedBimodal_h_
#define _RandomGeneratorCombinedBimodal_h_

#include  "RandomGeneratorCombined.h"
#include  "Bimodal.h"

class RandomGeneratorCombinedBimodal: public RandomGeneratorCombined, public Bimodal{
public:
  __host__ __device__ RandomGeneratorCombinedBimodal(Seed seed): RandomGeneratorCombined(seed), Bimodal(){};
  __host__ __device__ double GetRandomNumber();
};

#endif
