#include <cmath>
#include "Option.h"

__host__ __device__ Option::Option(OptionData OptionParameters, MontecarloPath* Path){
    _OptionParameters = OptionParameters;
    _Path = Path;
};

__host__ __device__ OptionForward::OptionForward(OptionData OptionParameters, MontecarloPath* Path):
    Option(OptionParameters, Path){};

__host__ __device__ double OptionForward::GetPayOff(){
    int NumberOfFixingDate=_Path->GetNumberOfFixingDate();
    double* Path=_Path->GetPath();

    return Path[NumberOfFixingDate-1];
};

__host__ __device__ OptionPlainVanilla::OptionPlainVanilla(OptionData OptionParameters, MontecarloPath* Path):
    Option(OptionParameters, Path){};

__host__ __device__  double OptionPlainVanilla::GetPayOff(){
    int NumberOfFixingDate=_Path->GetNumberOfFixingDate();
    double StrikePrice=_OptionParameters.AdditionalParameters[0];
    double* Path=_Path->GetPath();

    double PayOff=0;
    if( _OptionParameters.OptionType==1)
      PayOff=Path[NumberOfFixingDate-1]-StrikePrice;
    if( _OptionParameters.OptionType==2)
      PayOff=StrikePrice-Path[NumberOfFixingDate-1];

    if(PayOff>0) return PayOff;
    else return 0.;
};

__host__ __device__ OptionAbsolutePerformanceBarrier::OptionAbsolutePerformanceBarrier(OptionData OptionParameters, MontecarloPath* Path):
    Option(OptionParameters, Path){};

__host__ __device__  double OptionAbsolutePerformanceBarrier::GetPayOff(){
    int NumberOfFixingDate=_Path->GetNumberOfFixingDate();
    double Volatility=_Path->GetVolatility();
    double EquityInitialPrice=_Path->GetEquityInitialPrice();
    double N=_OptionParameters.AdditionalParameters[2];
    double B=_OptionParameters.AdditionalParameters[0];
    double K=_OptionParameters.AdditionalParameters[1];
    double* Path=_Path->GetPath();

    double SumP=0;
    double TStep= _OptionParameters.MaturityDate / NumberOfFixingDate;
    double Norm=1./sqrt(TStep);

    if( abs( Norm*log(Path[0]/EquityInitialPrice)) > B * Volatility )
        SumP=SumP+1.;

    for(int i=0; i<NumberOfFixingDate-1; i++){
        if( abs( Norm*log(Path[i+1]/Path[i]) ) > B * Volatility )
            SumP=SumP+1.;
    }

    if( SumP/NumberOfFixingDate-K > 0 )
        return N*(SumP/NumberOfFixingDate-K);
    else
        return 0;
};
