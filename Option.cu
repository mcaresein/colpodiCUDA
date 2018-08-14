#include <cmath>
#include "Option.h"

__host__ __device__ Option::Option(OptionData OptionParameters){
    _OptionParameters = OptionParameters;
};

__host__ __device__ OptionForward::OptionForward(OptionData OptionParameters):
    Option(OptionParameters){};

__host__ __device__ double OptionForward::GetPayOff(DatesVector Dates){
    double* Path=Dates.Path;
    int NumberOfFixingDate=Dates.NumberOfFixingDate;

    return Path[NumberOfFixingDate-1];
};

__host__ __device__ OptionPlainVanilla::OptionPlainVanilla(OptionData OptionParameters):
    Option(OptionParameters){};

__host__ __device__  double OptionPlainVanilla::GetPayOff(DatesVector Dates){
    double* Path=Dates.Path;
    int NumberOfFixingDate=Dates.NumberOfFixingDate;

    double StrikePrice=_OptionParameters.AdditionalParameters[0];

    double PayOff=0;
    if( _OptionParameters.OptionType==1)
      PayOff=Path[NumberOfFixingDate-1]-StrikePrice;
    if( _OptionParameters.OptionType==2)
      PayOff=StrikePrice-Path[NumberOfFixingDate-1];

    if(PayOff>0) return PayOff;
    else return 0.;
};

__host__ __device__ OptionAbsolutePerformanceBarrier::OptionAbsolutePerformanceBarrier(OptionData OptionParameters, double Volatility, double EquityInitialPrice):
    Option(OptionParameters){
        _Volatility=Volatility;
        _EquityInitialPrice=EquityInitialPrice;
    };

__host__ __device__  double OptionAbsolutePerformanceBarrier::GetPayOff(DatesVector Dates){
    double* Path=Dates.Path;
    int NumberOfFixingDate=Dates.NumberOfFixingDate;

    double N=_OptionParameters.AdditionalParameters[2];
    double B=_OptionParameters.AdditionalParameters[0];
    double K=_OptionParameters.AdditionalParameters[1];

    double SumP=0;
    double TStep= _OptionParameters.MaturityDate / NumberOfFixingDate;
    double Norm=1./sqrt(TStep);

    if( abs( Norm*log(Path[0]/_EquityInitialPrice)) > B * _Volatility )
        SumP=SumP+1.;

    for(int i=0; i<NumberOfFixingDate-1; i++){
        if( abs( Norm*log(Path[i+1]/Path[i]) ) > B * _Volatility )
            SumP=SumP+1.;
    }

    if( SumP/NumberOfFixingDate-K > 0 )
        return N*(SumP/NumberOfFixingDate-K);
    else
        return 0;
};
