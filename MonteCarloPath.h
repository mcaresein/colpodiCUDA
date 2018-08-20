/*#######################################################
# Classe che implementa la simulazione dell'evoluzione  #
# del prezzo del sottostante (GetPath)                  #
#######################################################*/

#ifndef _MonteCarloPath_h_
#define _MonteCarloPath_h_

#include "StochasticProcess.h"
#include "MarketData.h"
#include "DatesVector.h"
#include "UnderlyingPrice.h"

class MonteCarloPath{
public:
    __host__ __device__ MonteCarloPath(UnderlyingPrice* , double EquityInitialPrice, double MaturityDate, int NumberOfFixingDate, int EulerSubStep, bool AntitheticVariable);
    __host__ __device__ ~MonteCarloPath();
    __host__ __device__ DatesVector GetPath(StocasticProcess*);
    __host__ __device__ DatesVector GetAntitheticPath(StocasticProcess*);
private:
    double* _UnderlyingPath;
    double* _RandomNumbers;
    UnderlyingPrice* _Step;
    double _MaturityDate;
    double _EquityInitialPrice;
    int _NumberOfFixingDate;
    int _EulerSubStep;
    bool _AntitheticVariable;
};

#endif
