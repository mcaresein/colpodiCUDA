#include "Bimodal.h"

__host__ __device__ double Bimodal::GetBimodalRandomNumber(RandomGenerator* Generator){
  double RN=Generator->GetUniformRandomNumber();
  if (RN>=0.5){
     return 1.;
  }
  else{
     return -1.;
  }
};
