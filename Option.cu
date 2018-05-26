#include "Option.h"

__host__ __device__ Option::Option(OptionData OptionInput){
    _OptionInput = OptionInput;
};

__host__ __device__ int Option::GetNumberOfDatesToSimulate(){
    return _OptionInput.NumberOfDatesToSimulate;
};

__host__ __device__ double Option::GetTInitial(){
    return _OptionInput.TInitial;
};

__host__ __device__ double Option::GetMaturityDate(){
    return _OptionInput.MaturityDate;
};

__host__ __device__ OptionPlainVanillaCall::OptionPlainVanillaCall(OptionData OptionInput):
    Option(OptionInput){};

__host__ __device__  double OptionPlainVanillaCall::GetPayOff(double* OptionPath){

    double Difference=OptionPath[_OptionInput.NumberOfDatesToSimulate-1]-_OptionInput.StrikePrice;
    if(Difference>0) return Difference;
    else return 0.;
};

__host__ __device__ OptionPlainVanillaPut::OptionPlainVanillaPut(OptionData OptionInput):
    Option(OptionInput){};

__host__ __device__  double OptionPlainVanillaPut::GetPayOff(double* OptionPath){

    double Difference=_OptionInput.StrikePrice-OptionPath[_OptionInput.NumberOfDatesToSimulate-1];
    if(Difference>0) return Difference;
    else return 0.;
};
