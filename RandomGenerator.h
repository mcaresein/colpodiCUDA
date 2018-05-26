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
  __host__ __device__  virtual double GetGaussianRandomNumber();
protected:
  bool _Status, _GPU;
  double _SavedRandomNumber;
};

class CombinedGenerator: public RandomGenerator{
public:
  __host__ __device__  CombinedGenerator(Seed, bool);
  __host__ __device__  unsigned int LCGStep(unsigned int &, unsigned int , unsigned long );
  __host__ __device__  unsigned int TausStep(unsigned int &, unsigned int , unsigned int , unsigned int , unsigned long );
  __host__ __device__  double GetUniformRandomNumber();
private:
  unsigned int _Sa, _Sb, _Sc, _Sd;
};

#endif
