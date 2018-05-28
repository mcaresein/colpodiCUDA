#include <iostream>
#include <cmath>
#include "UnderlyingPath.h"
#include "RandomGenerator.h"
#include "StocasticProcess.h"
#include "Option.h"

__host__ __device__  MontecarloPath::MontecarloPath(double EquityInitialPrice, double MaturityDate, double TInitial, int NumberOfDatesToSimulate ,RandomGenerator* Generator,StocasticProcess* Process){
    _EquityInitialPrice = EquityInitialPrice;
    _Generator = Generator;
    _Process = Process;
    _MaturityDate=MaturityDate;
    _TInitial=TInitial;
    _NumberOfDatesToSimulate=NumberOfDatesToSimulate;
    _UnderlyingPath = new double[NumberOfDatesToSimulate];
};

__host__ __device__  MontecarloPath::~MontecarloPath(){
    delete[] _UnderlyingPath;
};

__host__ __device__  double* MontecarloPath::GetPath(){

    double TStep = ( _MaturityDate - _TInitial ) /  _NumberOfDatesToSimulate ;
    double temp=_EquityInitialPrice;

    for(int i=0; i<_NumberOfDatesToSimulate; i++){
        double w = _Generator->GetGaussianRandomNumber();
        temp = _Process->Step(temp, TStep, w);
        _UnderlyingPath[i]=temp;
    }

    return _UnderlyingPath;

};
