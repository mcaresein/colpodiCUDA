#include "RandomGeneratorCombinedBinomial.h"

__host__ __device__ void RandomGeneratorCombinedBinomial::SetRandomNumber(){
  _RandomNumber=this->GetBinomialRandomNumber(this);
};
