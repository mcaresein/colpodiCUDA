#include <cmath>
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

__host__ __device__ OptionForward::OptionForward(OptionData OptionInput):
    Option(OptionInput){};

__host__ __device__ double OptionForward::GetPayOff(double* OptionPath){
    return OptionPath[_OptionInput.NumberOfDatesToSimulate-1];
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

__host__ __device__ OptionAbsolutePerformanceBarrier::OptionAbsolutePerformanceBarrier(OptionData OptionInput, double volatility):
    Option(OptionInput){
        _Volatility=volatility;
    };

__host__ __device__  double OptionAbsolutePerformanceBarrier::GetPayOff(double* OptionPath){
    double SumP=0;
    double TStep=( _OptionInput.MaturityDate - _OptionInput.TInitial ) /  _OptionInput.NumberOfDatesToSimulate;
    double Norm=1./sqrt(TStep);
    for(int i=0; i<_OptionInput.NumberOfDatesToSimulate-1; i++){
        if( abs( Norm*log(OptionPath[i+1]/OptionPath[i]) ) > _OptionInput.B * _Volatility )
            SumP=SumP+1.;
    }
    if( SumP/_OptionInput.NumberOfDatesToSimulate-_OptionInput.K > 0 )
        return _OptionInput.N*(SumP/_OptionInput.NumberOfDatesToSimulate-_OptionInput.K);
    else
        return 0;
};
