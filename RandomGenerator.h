/*#########################################################
# Classe per la generazione di numeri casuali distribuiti #
# uniformemente (Uniform) o gaussianamente (Gauss).       #
#########################################################*/

#ifndef _RandomGenerator_h_
#define _RandomGenerator_h_

#include "DataTypes.h"

class RandomGenerator{
public:
  __host__ __device__  virtual float Uniform()=0;
  __host__ __device__  float Gauss();
};

class CombinedGenerator: public RandomGenerator{
public:
  __host__ __device__  CombinedGenerator(Seed);
  __host__ __device__  unsigned int LCGStep(unsigned int seed, unsigned int a, unsigned long b);
  __host__ __device__  unsigned int TausStep(unsigned int seed, unsigned int K1, unsigned int K2, unsigned int K3, unsigned long M);
  __host__ __device__  float Uniform();
private:
  unsigned int _Sa, _Sb, _Sc, _Sd;
};

#endif
