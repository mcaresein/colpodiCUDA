#include "RandomGeneratorCombinedGaussian.h"

__host__ __device__ double RandomGeneratorCombinedGaussian::GetRandomNumber(){
  return this->GetGaussianRandomNumber(this);
};
