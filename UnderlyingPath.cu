#include <iostream>
#include <cmath>
#include "UnderlyingPath.h"
#include "RandomGenerator.h"
#include "StocasticProcess.h"

__host__ __device__  MontecarloPath::MontecarloPath(double SInitial, double TInitial, double TFinal,RandomGenerator* Generator,StocasticProcess* Process, int NSteps){
    _TInitial = TInitial;
    _SInitial = SInitial;
    _Generator = Generator;
    _Process = Process;
    _UnderlyingPath = new double[NSteps];
    _TFinal=TFinal;
    _NSteps=NSteps;
};

__host__ __device__  MontecarloPath::~MontecarloPath(){
    delete[] _UnderlyingPath;
};

__host__ __device__  double* MontecarloPath::GetPath(){

    double TStep = (_TFinal-_TInitial)/_NSteps;
    double temp=_SInitial;

    for(int i=0; i<_NSteps; i++){
        double w = _Generator->Gauss();
        temp = _Process->Step(temp, TStep, w);
        _UnderlyingPath[i]=temp;
    }

    return _UnderlyingPath;

};
