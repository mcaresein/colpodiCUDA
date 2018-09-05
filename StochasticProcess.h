/*####################################################################################################
# Classi che implementano i metodi per simulare gli step temporali dell'evoluzione del prezzo del    #
# sottostante secondo processo esatto (ExactLogNormalProcess) o approssimato (EulerLogNormalProcess) #
####################################################################################################*/

#ifndef _StochasticProcess_h_
#define _StochasticProcess_h_

#include "RandomGenerator.h"
#include "RandomGeneratorCombined.h"
#include "MarketData.h"
#include "UnderlyingPrice.h"

class StochasticProcess{
public:
    __host__ __device__ virtual void Step(UnderlyingPrice*, double TimeStep, double RandomNumber)=0;
    __host__ __device__ RandomGenerator* GetRandomGenerator();
    __host__ __device__ double GetRandomNumber();
protected:
    RandomGenerator* _Generator;
};

class ExactLogNormalProcess: public StochasticProcess{
public:
    __host__ __device__ ExactLogNormalProcess(RandomGenerator* Generator);
    __host__ __device__ void Step(UnderlyingPrice * Step, double TimeStep, double RandomNumber);
};

class EulerLogNormalProcess: public StochasticProcess{
public:
    __host__ __device__ EulerLogNormalProcess(RandomGenerator* Generator);
    __host__ __device__ void Step(UnderlyingPrice * Step, double TimeStep,  double RandomNumber);
};

#endif
