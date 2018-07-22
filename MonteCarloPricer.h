/*#####################################################
# Classe che restituisce la somma e la somma quadrata #
# dei Pay Off calcolati in uno stream (ComputePrice). #
#####################################################*/

#ifndef _MonteCarloPricer_h_
#define _MonteCarloPricer_h_

#include "Option.h"
#include "Statistics.h"

class MonteCarloPricer{
public:
    __device__ __host__ MonteCarloPricer(Option*, int);
    __device__ __host__ void ComputePrice(Statistics*);
private:
    Option* _Option;
    int _NStreams;
};

#endif
