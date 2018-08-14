/*########################################################################
# Classe per la generazione di numeri casuali distribuiti                #
# uniformemente (GetUniformRandomNumber) o gaussianamente (GetGaussianRandomNumber).       #
########################################################################*/

#ifndef _RandomGenerator_h_
#define _RandomGenerator_h_

class RandomGenerator{
public:
    __host__ __device__  virtual double GetUniformRandomNumber()=0;
    __host__ __device__  virtual double GetGaussianRandomNumber();
protected:
    double _SavedRandomNumber;
    bool _Status, _BoxMullerWithReExtraction;
};

#endif
