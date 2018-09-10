#include "MonteCarloPricer.h"

__device__ __host__ MonteCarloPricer::MonteCarloPricer(Option* Option, MonteCarloPath* Path, StochasticProcess* Process, int NStreams, bool AntitheticVariable){
    _NStreams=NStreams;
    _Option=Option;
    _Path=Path;
    _Process=Process;
    _AntitheticVariable=AntitheticVariable;
};

//## Metodo per il calcolo delle somme semplici e quadrate dei PayOff simulati in uno stream. ##

__device__ __host__ void MonteCarloPricer::ComputePrice(Statistics* PayOffs){
    for(int j=0; j<_NStreams; j++){
        DatesVector Dates=_Path->GetPath(_Process);
        double PayOff=_Option->GetPayOff(Dates);
        if(_AntitheticVariable==false)
            PayOffs->AddValue(PayOff);
        else{
            DatesVector AntitheticDates=_Path->GetAntitheticPath(_Process);
            double AntitheticPayOff=_Option->GetPayOff(AntitheticDates);
            PayOffs->AddValue(0.5*(PayOff+AntitheticPayOff));
        }
    }
};
