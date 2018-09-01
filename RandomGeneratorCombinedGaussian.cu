#include "RandomGeneratorCombinedGaussian.h"

__host__ __device__ double RandomGeneratorCombinedGaussian::GetRandomVariable(){
  return this->GetGaussianRandomNumber(this->GetUniformRandomNumber(), this->GetUniformRandomNumber());
};
