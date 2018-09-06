#include "RandomGeneratorCombinedGaussian.h"
#include <iostream>
__host__ __device__ double RandomGeneratorCombinedGaussian::GetRandomNumber(){
  return this->GetGaussianRandomNumber(this);
};
