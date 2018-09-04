#include <cmath>
#include "StochasticProcess.h"

__host__ __device__ ExactLogNormalProcess::ExactLogNormalProcess(RandomGenerator* Generator){
    _Generator=Generator;
};
__host__ __device__ RandomGenerator* StochasticProcess::GetRandomGenerator(){
    return _Generator;
};
__host__ __device__ void ExactLogNormalProcess::Step(UnderlyingPrice * Step, double TimeStep, double RandomNumber){
    double Drift=Step->Anagraphy->Drift;
    double Volatility=Step->Anagraphy->Volatility;
    Step->Price=Step->Price*exp((Drift - (Volatility*Volatility)/2)*TimeStep + Volatility*sqrt(TimeStep)*RandomNumber);

};


__host__ __device__  EulerLogNormalProcess::EulerLogNormalProcess(RandomGenerator* Generator){
    _Generator=Generator;
};

__host__ __device__ void EulerLogNormalProcess::Step(UnderlyingPrice * Step, double TimeStep,  double RandomNumber){
    double Drift=Step->Anagraphy->Drift;
    double Volatility=Step->Anagraphy->Volatility;
    Step->Price=Step->Price*(1.+Drift*TimeStep+Volatility*sqrt(TimeStep)*RandomNumber);
};
