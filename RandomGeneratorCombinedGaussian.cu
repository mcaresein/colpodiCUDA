#include "RandomGeneratorCombinedGaussian.h"
#include <iostream>
__host__ __device__ void RandomGeneratorCombinedGaussian::SetRandomNumber(){
  _RandomNumber=this->GetGaussianRandomNumber(this);
};
