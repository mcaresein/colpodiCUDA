#include "PayOff.h"

__host__ __device__  double PayOffPlainVanillaCall::GetPayOff(double* OptionPath, double StrikePrice, int NSteps){

    double Difference=OptionPath[NSteps-1]-StrikePrice;
    if(Difference>0) return Difference;
    else return 0.;
};

__host__ __device__  double PayOffPlainVanillaPut::GetPayOff(double* OptionPath, double StrikePrice, int NSteps){

    double Difference=StrikePrice-OptionPath[NSteps-1];
    if(Difference>0) return Difference;
    else return 0.;
};
