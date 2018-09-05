#include "MonteCarloPath.h"
#include <iostream>
//new
__host__ __device__  MonteCarloPath::MonteCarloPath(UnderlyingPrice* Price, double EquityInitialPrice, double MaturityDate, int NumberOfFixingDate, int EulerSubStep, bool AntitheticVariable){
    _MaturityDate=MaturityDate;
    _NumberOfFixingDate=NumberOfFixingDate;
    _UnderlyingPath = new double[NumberOfFixingDate];
    _EulerSubStep = EulerSubStep;
    _Step=Price;
    _EquityInitialPrice=EquityInitialPrice;
    _AntitheticVariable=AntitheticVariable;
    if(AntitheticVariable==true)
        _RandomNumbers= new double[NumberOfFixingDate*EulerSubStep];
};

__host__ __device__  MonteCarloPath::~MonteCarloPath(){
    delete[] _UnderlyingPath;
    if(_AntitheticVariable==true)
        delete[] _RandomNumbers;
};
//new
__host__ __device__  DatesVector MonteCarloPath::GetPath(StochasticProcess* Process){
    double TimeStep =  _MaturityDate / (_NumberOfFixingDate*_EulerSubStep);
    _Step->Price=_EquityInitialPrice;

    for(int i=0; i<_NumberOfFixingDate; i++){
        for(int j=0; j<_EulerSubStep; j++){
          Process->GetRandomGenerator()->SetRandomNumber();
          double rnd=Process->GetRandomNumber();
          Process->Step(_Step, TimeStep, rnd);
          if(_AntitheticVariable==true)
            _RandomNumbers[i*_EulerSubStep+j]=rnd;
        }
        _UnderlyingPath[i]=_Step->Price;
    }

    DatesVector Dates;
    Dates.Path=_UnderlyingPath;
    Dates.NumberOfFixingDate=_NumberOfFixingDate;
    return Dates;
};

__host__ __device__  DatesVector MonteCarloPath::GetAntitheticPath(StochasticProcess* Process){
    double TimeStep =  _MaturityDate / (_NumberOfFixingDate*_EulerSubStep);
    _Step->Price=_EquityInitialPrice;
    for(int i=0; i<_NumberOfFixingDate; i++){
        for(int j=0; j<_EulerSubStep; j++){
            Process->Step(_Step, TimeStep, -1.*_RandomNumbers[i*_EulerSubStep+j]);
        }
        _UnderlyingPath[i]=_Step->Price;
    }

    DatesVector Dates;
    Dates.Path=_UnderlyingPath;
    Dates.NumberOfFixingDate=_NumberOfFixingDate;
    return Dates;
}
