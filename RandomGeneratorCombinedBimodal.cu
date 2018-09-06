#include "RandomGeneratorCombinedBimodal.h"

__host__ __device__ double RandomGeneratorCombinedBimodal::GetRandomNumber(){
  return this->GetBimodalRandomNumber(this);
};
