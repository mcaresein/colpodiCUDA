#include <iostream>
#include <cmath>
#include "RandomGenerator.h"
#include "RandomGeneratorCombined.h"
#include "Seed.h"

__host__ __device__  RandomGeneratorCombined::RandomGeneratorCombined(Seed S){
    _Seed=S;
};

__host__ __device__  unsigned int RandomGeneratorCombined::LCGStep(unsigned int &seed, unsigned int a, unsigned long b){
	return seed=(a*seed+b)%UINT_MAX;

};

__host__ __device__  unsigned int RandomGeneratorCombined::TausStep(unsigned int &seed, unsigned int K1, unsigned int K2, unsigned int K3, unsigned long M){
	unsigned int b=(((seed<<K1)^seed)>>K2);
  return seed=(((seed&M)<<K3)^b);

};

__host__ __device__  double RandomGeneratorCombined::GetUniformRandomNumber(){
    return 2.3283064365387e-10*(TausStep(_Seed.S1, 13, 19, 12, 4294967294UL)^TausStep(_Seed.S2, 2, 25, 4, 4294967288UL)^TausStep(_Seed.S3, 3, 11, 17, 4294967280UL)^LCGStep(_Seed.S4, 1664525, 1013904223UL));
};
