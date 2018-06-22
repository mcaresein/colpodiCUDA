#include <cmath>
#include "Option.h"

__host__ __device__ Option::Option(OptionData OptionInput, MontecarloPath* Path){
    _OptionInput = OptionInput;
    _Path = Path;
};

__host__ __device__ int Option::GetNumberOfDatesToSimulate(){
    return _OptionInput.NumberOfDatesToSimulate;
};

//__host__ __device__ int Option::GetEulerSubStep(){
//    return _OptionInput.EulerSubStep;
//};

//__host__ __device__ double Option::GetMaturityDate(){
//    return _OptionInput.MaturityDate;
//};

__host__ __device__ OptionForward::OptionForward(OptionData OptionInput, MontecarloPath* Path):
    Option(OptionInput, Path){};

__host__ __device__ double OptionForward::GetPayOff(double* OptionPath){
    return OptionPath[_OptionInput.NumberOfDatesToSimulate-1];
};

__device__ __host__ MontecarloPath* Option::GetMontecarloPath(){
    return _Path;
};

__host__ __device__ OptionPlainVanilla::OptionPlainVanilla(OptionData OptionInput, MontecarloPath* Path):
    Option(OptionInput, Path){};

__host__ __device__  double OptionPlainVanilla::GetPayOff(double* OptionPath){

    double Difference=0;
    if(_OptionInput.OptionTypeCallOrPut==1)
      Difference=OptionPath[_OptionInput.NumberOfDatesToSimulate-1]-_OptionInput.StrikePrice;
    if(_OptionInput.OptionTypeCallOrPut==2)
      Difference=_OptionInput.StrikePrice-OptionPath[_OptionInput.NumberOfDatesToSimulate-1];

    if(Difference>0) return Difference;
    else return 0.;

};

__host__ __device__ OptionAbsolutePerformanceBarrier::OptionAbsolutePerformanceBarrier(OptionData OptionInput, MontecarloPath* Path, double volatility, double InitialPrice):
    Option(OptionInput, Path){
        _Volatility=volatility;
        _InitialPrice=InitialPrice;
    };

__host__ __device__  double OptionAbsolutePerformanceBarrier::GetPayOff(double* OptionPath){
    double SumP=0;
    double TStep= _OptionInput.MaturityDate /  _OptionInput.NumberOfDatesToSimulate;
    double Norm=1./sqrt(TStep);

    if( abs( Norm*log(OptionPath[0]/_InitialPrice) ) > _OptionInput.B * _Volatility )
        SumP=SumP+1.;

    for(int i=0; i<_OptionInput.NumberOfDatesToSimulate-1; i++){
        if( abs( Norm*log(OptionPath[i+1]/OptionPath[i]) ) > _OptionInput.B * _Volatility )
            SumP=SumP+1.;
    }
    if( SumP/_OptionInput.NumberOfDatesToSimulate-_OptionInput.K > 0 )
        return _OptionInput.N*(SumP/_OptionInput.NumberOfDatesToSimulate-_OptionInput.K);
    else
        return 0;
};
