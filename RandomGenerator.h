/*########################################################################
# Classe per la generazione di numeri casuali distribuiti                #
# uniformemente (GetUniformRandomNumber) o gaussianamente (GetGaussianRandomNumber).       #
########################################################################*/

#ifndef _RandomGenerator_h_
#define _RandomGenerator_h_

#include "Seed.h"

class RandomGenerator{
public:
    __host__ __device__  virtual double GetUniformRandomNumber()=0;
    __host__ __device__  virtual double GetRandomVariable();
protected:
    double _RandomVariable;
};

#endif
