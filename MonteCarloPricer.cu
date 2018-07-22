#include "MonteCarloPricer.h"

__device__ __host__ MonteCarloPricer::MonteCarloPricer(Option* Option, int NStreams){
    _NStreams=NStreams;
    _Option=Option;
};

//## Metodo per il calcolo delle somme semplici e quadrate dei PayOff simulati in uno stream. ##

__device__ __host__ void MonteCarloPricer::ComputePrice(Statistics* PayOffs){
    for(int j=0; j<_NStreams; j++)
        PayOffs->AddValue(_Option->GetPayOff());
};
