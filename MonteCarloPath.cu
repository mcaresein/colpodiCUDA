#include "MonteCarloPath.h"

__host__ __device__  MonteCarloPath::MonteCarloPath(UnderlyingPrice* Price, double EquityInitialPrice, double MaturityDate, int NumberOfFixingDate, int EulerSubStep){
    _MaturityDate=MaturityDate;
    _NumberOfFixingDate=NumberOfFixingDate;
    _UnderlyingPath = new double[NumberOfFixingDate];
    _RandomNumbers= new double[NumberOfFixingDate*EulerSubStep];
    _EulerSubStep = EulerSubStep;
    _Step=Price;
    _EquityInitialPrice=EquityInitialPrice;
};

__host__ __device__  MonteCarloPath::~MonteCarloPath(){
    delete[] _UnderlyingPath;
};

__host__ __device__  DatesVector MonteCarloPath::GetPath(StocasticProcess* Process){
    double TimeStep =  _MaturityDate / (_NumberOfFixingDate*_EulerSubStep);
    _Step->Price=_EquityInitialPrice;

    for(int i=0; i<_NumberOfFixingDate*_EulerSubStep; i++)
        _RandomNumbers[i]=Process->GetRandomNumber();

    for(int i=0; i<_NumberOfFixingDate; i++){
        for(int j=0; j<_EulerSubStep; j++){
            Process->Step(_Step, TimeStep, _RandomNumbers[i*_EulerSubStep+j]);
        }
        _UnderlyingPath[i]=_Step->Price;
    }

    DatesVector Dates;
    Dates.Path=_UnderlyingPath;
    Dates.NumberOfFixingDate=_NumberOfFixingDate;
    return Dates;
};

__host__ __device__  DatesVector MonteCarloPath::GetAntitheticPath(StocasticProcess* Process){
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
