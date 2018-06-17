#include <iostream>
#include <cmath>
#include "UnderlyingPath.h"
#include "StocasticProcess.h"
#include "Option.h"

__host__ __device__  MontecarloPath::MontecarloPath(MarketData MarketInput, double MaturityDate, int NumberOfDatesToSimulate ,StocasticProcess* Process, int EulerSubStep){
    _MarketInput=MarketInput;
    _Process = Process;
    _MaturityDate=MaturityDate;
    _NumberOfDatesToSimulate=NumberOfDatesToSimulate;
    _UnderlyingPath = new double[NumberOfDatesToSimulate];
    _EulerSubStep = EulerSubStep;
};

__host__ __device__  MontecarloPath::~MontecarloPath(){
    delete[] _UnderlyingPath;
};

__host__ __device__  double* MontecarloPath::GetPath(){

    double TStep =  _MaturityDate /  (_NumberOfDatesToSimulate*_EulerSubStep) ;
    double temp = _MarketInput.EquityInitialPrice;

    for(int i=0; i<_NumberOfDatesToSimulate; i++){
        for(int j=0; j<_EulerSubStep; j++){
            temp = _Process->Step(_MarketInput, TStep, temp);
        }
        _UnderlyingPath[i]=temp;
    }

    return _UnderlyingPath;

};
