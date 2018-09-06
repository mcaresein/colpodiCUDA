#include "RandomGeneratorCombinedBinomial.h"

__host__ __device__ double RandomGeneratorCombinedBinomial::GetRandomNumber(){
  return this->GetBinomialRandomNumber(this);
};
