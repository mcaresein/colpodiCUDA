/*####################################################################################################
# Classi che implementano i metodi per simulare gli step temporali dell'evoluzione del prezzo del    #
# sottostante secondo processo esatto (ExactLogNormalProcess) o approssimato (EulerLogNormalProcess) #
####################################################################################################*/

#ifndef _StocasticProcess_h_
#define _StocasticProcess_h_

#include "RandomGenerator.h"
#include "MarketData.h"

class StocasticProcess{
public:
    __host__ __device__  virtual double Step(MarketData , double TimeStep, double PriceStep)=0;
};

class ExactLogNormalProcess: public StocasticProcess{
public:
    __host__ __device__  ExactLogNormalProcess(RandomGenerator* Generator);
    __host__ __device__  double Step(MarketData , double TimeStep, double PriceStep);
private:
    RandomGenerator* _Generator;
};

class EulerLogNormalProcess: public StocasticProcess{
public:
    __host__ __device__  EulerLogNormalProcess(RandomGenerator* Generator);
    __host__ __device__  double Step(MarketData , double TimeStep, double PriceStep);
private:
    RandomGenerator* _Generator;
};

#endif
