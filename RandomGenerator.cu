#include <iostream>
#include <cmath>
#include "RandomGenerator.h"
#include "Seed.h"

__host__ __device__ double RandomGenerator::GetRandomNumber(){
  return _RandomNumber;
};
