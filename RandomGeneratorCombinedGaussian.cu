#include "RandomGeneratorCombinedGaussian.h"

__host__ __device__ void RandomGeneratorCombinedGaussian::SetRandomNumber(){
  _RandomNumber=this->GetGaussianRandomNumber(this);
};
