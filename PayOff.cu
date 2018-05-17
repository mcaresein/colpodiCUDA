#include "PayOff.h"

__host__ __device__  float PayOffPlainVanilla::GetPayOff(float* OptionPath, float StrikePrice, int NSteps){

    float Difference=OptionPath[NSteps-1]-StrikePrice;
    if(Difference>0) return Difference;
    else return 0.;
};
