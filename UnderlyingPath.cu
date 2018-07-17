#include <iostream>
#include <cmath>
#include "UnderlyingPath.h"
#include "StocasticProcess.h"
#include "Option.h"

__host__ __device__  MontecarloPath::MontecarloPath(MarketData MarketInput, double MaturityDate, int NumberOfFixingDate ,StocasticProcess* Process, int EulerSubStep){
    _MarketInput=MarketInput;
    _Process = Process;
    _MaturityDate=MaturityDate;
    _NumberOfFixingDate=NumberOfFixingDate;
    _UnderlyingPath = new double[NumberOfFixingDate];
    _EulerSubStep = EulerSubStep;
};

__host__ __device__  MontecarloPath::~MontecarloPath(){
    delete[] _UnderlyingPath;
};

__host__ __device__  double* MontecarloPath::GetPath(){

    double TStep =  _MaturityDate /  (_NumberOfFixingDate*_EulerSubStep) ;
    double temp = _MarketInput.EquityInitialPrice;

    for(int i=0; i<_NumberOfFixingDate; i++){
        for(int j=0; j<_EulerSubStep; j++){
            temp = _Process->Step(_MarketInput, TStep, temp);
        }
        _UnderlyingPath[i]=temp;
    }

    return _UnderlyingPath;

};
__host__ __device__  MarketData MontecarloPath::GetMarketData(){
    return _MarketInput;
};
