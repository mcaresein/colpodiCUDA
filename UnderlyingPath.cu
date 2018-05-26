#include <iostream>
#include <cmath>
#include "UnderlyingPath.h"
#include "RandomGenerator.h"
#include "StocasticProcess.h"
#include "Option.h"

__host__ __device__  MontecarloPath::MontecarloPath(double EquityInitialPrice, Option* Option ,RandomGenerator* Generator,StocasticProcess* Process){
    _EquityInitialPrice = EquityInitialPrice;
    _Generator = Generator;
    _Process = Process;
    _Option = Option;
    _UnderlyingPath = new double[Option->GetNumberOfDatesToSimulate()];
};

__host__ __device__  MontecarloPath::~MontecarloPath(){
    delete[] _UnderlyingPath;
};

__host__ __device__  double* MontecarloPath::GetPath(){

    double TStep = ( _Option->GetMaturityDate() - _Option->GetTInitial() ) / ( _Option->GetNumberOfDatesToSimulate() );
    double temp=_EquityInitialPrice;

    for(int i=0; i<_Option->GetNumberOfDatesToSimulate(); i++){
        double w = _Generator->GetGaussianRandomNumber();
        temp = _Process->Step(temp, TStep, w);
        _UnderlyingPath[i]=temp;
    }

    return _UnderlyingPath;

};
