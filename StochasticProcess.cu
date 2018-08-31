#include <cmath>
#include "StochasticProcess.h"

__host__ __device__ ExactLogNormalProcess::ExactLogNormalProcess(RandomGenerator* Generator){
    _Generator=Generator;
};
__host__ __device__ double StochasticProcess::GetRandomNumber(){
    return _Generator->GetRandomVariable();
};
__host__ __device__ void ExactLogNormalProcess::Step(UnderlyingPrice * Step, double TimeStep){
    double Drift=Step->Anagraphy->Drift;
    double Volatility=Step->Anagraphy->Volatility;
    Step->Price=Step->Price*exp((Drift - (Volatility*Volatility)/2)*TimeStep + Volatility*sqrt(TimeStep)*_Generator->GetRandomVariable());

};


__host__ __device__  EulerLogNormalProcess::EulerLogNormalProcess(RandomGenerator* Generator){
    _Generator=Generator;
};

__host__ __device__ void EulerLogNormalProcess::Step(UnderlyingPrice * Step, double TimeStep){
    double Drift=Step->Anagraphy->Drift;
    double Volatility=Step->Anagraphy->Volatility;
    Step->Price=Step->Price*(1.+Drift*TimeStep+Volatility*sqrt(TimeStep)*_Generator->GetRandomVariable());
};
