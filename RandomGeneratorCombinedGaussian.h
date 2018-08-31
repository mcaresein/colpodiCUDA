#ifndef _RandomGeneratorCombinedGaussian_h_
#define _RandomGeneratorCombinedGaussian_h_

#include  "RandomGeneratorCombined.h"
#include  "Gaussian.h"

class RandomGeneratorCombinedGaussian: public RandomGeneratorCombined, public Gaussian{
public:
    __host__ __device__  RandomGeneratorCombinedGaussian(Seed seed, bool BoxMullerWithReExtraction): RandomGeneratorCombined(seed), Gaussian(BoxMullerWithReExtraction){};
    __host__ __device__  double GetRandomVariable();
};

#endif
