/*#########################################################
# Classe per la generazione di numeri casuali distribuiti #
# uniformemente (Uniform) o gaussianamente (Gauss).       #
#########################################################*/

#ifndef _RandomGenerator_h_
#define _RandomGenerator_h_

#include "DataTypes.h"

class RandomGenerator{
public:
  __host__ __device__  virtual double Uniform()=0;
  __host__ __device__  double Gauss();
};

class CombinedGenerator: public RandomGenerator{
public:
  __host__ __device__  CombinedGenerator(Seed);
  __host__ __device__  unsigned int LCGStep(unsigned int &, unsigned int , unsigned long );
  __host__ __device__  unsigned int TausStep(unsigned int &, unsigned int , unsigned int , unsigned int , unsigned long );
  __host__ __device__  double Uniform();
private:
  unsigned int _Sa, _Sb, _Sc, _Sd;
};

#endif
