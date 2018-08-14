#include "MonteCarloPricer.h"

__device__ __host__ MonteCarloPricer::MonteCarloPricer(Option* Option, MonteCarloPath* Path, StocasticProcess* Process, int NStreams, bool AntitheticVariable){
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
        PayOffs->AddValue(_Option->GetPayOff(Dates));
        if(_AntitheticVariable==true){
            DatesVector AntitheticDates=_Path->GetAntitheticPath(_Process);
            PayOffs->AddValue(_Option->GetPayOff(AntitheticDates));
        }
    }
};
