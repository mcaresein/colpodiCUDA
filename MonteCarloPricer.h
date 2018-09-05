/*#####################################################
# Classe che restituisce la somma e la somma quadrata #
# dei Pay Off calcolati in uno stream (ComputePrice). #
#####################################################*/

#ifndef _MonteCarloPricer_h_
#define _MonteCarloPricer_h_

#include "Option.h"
#include "Statistics.h"
#include "MonteCarloPath.h"
#include "StochasticProcess.h"
#include "DatesVector.h"

class MonteCarloPricer{
public:
    __device__ __host__ MonteCarloPricer(Option*, MonteCarloPath*, StochasticProcess*, int Nstreams, bool AntitheticVariable);
    __device__ __host__ void ComputePrice(Statistics*);
private:
    Option* _Option;
    MonteCarloPath* _Path;
    StochasticProcess* _Process;
    int _NStreams;
    bool _AntitheticVariable;
};

#endif
