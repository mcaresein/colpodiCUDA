/*########################################################################
# Classe per la generazione di numeri casuali distribuiti                #
# uniformemente (GetUniformRandomNumber) o gaussianamente (GetGaussianRandomNumber).       #
########################################################################*/

#ifndef _RandomGenerator_h_
#define _RandomGenerator_h_

#include "DataTypes.h"

class RandomGenerator{
public:
  __host__ __device__  virtual double GetUniformRandomNumber()=0;
  __host__ __device__  virtual double GetGaussianRandomNumber()=0;
};

class CombinedGenerator: public RandomGenerator{
public:
  __host__ __device__  CombinedGenerator(Seed, bool);
  __host__ __device__  unsigned int LCGStep(unsigned int &, unsigned int , unsigned long );
  __host__ __device__  unsigned int TausStep(unsigned int &, unsigned int , unsigned int , unsigned int , unsigned long );
  __host__ __device__  double GetUniformRandomNumber();
  __host__ __device__  double GetGaussianRandomNumber();
protected:
  unsigned int _Sa, _Sb, _Sc, _Sd;
  double _SavedRandomNumber;
  bool _Status, _GPU;
};

#endif
