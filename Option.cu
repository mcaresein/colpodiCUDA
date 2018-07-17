#include <cmath>
#include <iostream>
#include "Option.h"

__host__ __device__ Option::Option(OptionData OptionParameters, MontecarloPath* Path){
    _OptionParameters = OptionParameters;
    _Path = Path;
};

__host__ __device__ int Option::GetNumberOfFixingDate(){
    return _OptionParameters.NumberOfFixingDate;
};

__host__ __device__ OptionForward::OptionForward(OptionData OptionParameters, MontecarloPath* Path):
    Option(OptionParameters, Path){};

__host__ __device__ double OptionForward::GetPayOff(double* OptionPath, int NumberOfFixingDate){
    return OptionPath[NumberOfFixingDate-1];
};

__device__ __host__ MontecarloPath* Option::GetMontecarloPath(){
    return _Path;
};

__host__ __device__ OptionPlainVanilla::OptionPlainVanilla(OptionData OptionParameters, MontecarloPath* Path):
    Option(OptionParameters, Path){};

__host__ __device__  double OptionPlainVanilla::GetPayOff(double* OptionPath, int NumberOfFixingDate){

    double Difference=0;
    if( _OptionParameters.OptionType==1)
      Difference=OptionPath[NumberOfFixingDate-1]-_OptionParameters.AdditionalParameters[0];
    if( _OptionParameters.OptionType==2)
      Difference=_OptionParameters.AdditionalParameters[0]-OptionPath[NumberOfFixingDate-1];

    if(Difference>0) return Difference;
    else return 0.;

};

__host__ __device__ OptionAbsolutePerformanceBarrier::OptionAbsolutePerformanceBarrier(OptionData OptionParameters, MontecarloPath* Path):
    Option(OptionParameters, Path){
        _Volatility=(Path->GetMarketData()).Volatility;
        _InitialPrice=Path->GetMarketData().EquityInitialPrice;
    };

__host__ __device__  double OptionAbsolutePerformanceBarrier::GetPayOff(double* OptionPath, int NumberOfFixingDate){
    double SumP=0;
    double TStep= _OptionParameters.MaturityDate / NumberOfFixingDate;
    double Norm=1./sqrt(TStep);

    if( abs( Norm*log(OptionPath[0]/_InitialPrice) ) > _OptionParameters.AdditionalParameters[0] * _Volatility )
        SumP=SumP+1.;

    for(int i=0; i<NumberOfFixingDate-1; i++){
        if( abs( Norm*log(OptionPath[i+1]/OptionPath[i]) ) > _OptionParameters.AdditionalParameters[0] * _Volatility )
            SumP=SumP+1.;
    }

    if( SumP/NumberOfFixingDate-_OptionParameters.AdditionalParameters[1] > 0 )
        return _OptionParameters.AdditionalParameters[2]*(SumP/NumberOfFixingDate-_OptionParameters.AdditionalParameters[1]);
    else
        return 0;
};
