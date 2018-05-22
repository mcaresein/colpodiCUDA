/*#######################################################
# Classe che implementa la simulazione dell'evoluzione  #
# del prezzo del sottostante (GetPath)                  #
#######################################################*/

#ifndef _UnderlyingPath_h_
#define _UnderlyingPath_h_

#include "RandomGenerator.h"
#include "StocasticProcess.h"

class UnderlyingPath{
public:
  __host__ __device__  virtual double* GetPath()=0;
};

class MontecarloPath: public UnderlyingPath{
public:
  __host__ __device__  MontecarloPath(double, double, double,RandomGenerator*, StocasticProcess*, int);
  __host__ __device__  ~MontecarloPath();
  __host__ __device__  double* GetPath();

private:
    double* _UnderlyingPath;
    RandomGenerator* _Generator;
    StocasticProcess* _Process;
    double _TInitial;
    double _SInitial;
    double _TFinal;
    int _NSteps;
};

#endif
