/*#######################################################
# Classe che implementa la simulazione dell'evoluzione  #
# del prezzo del sottostante (GetPath)                  #
#######################################################*/

#ifndef _UnderlyingPath_h_
#define _UnderlyingPath_h_

#include "StocasticProcess.h"
//#include "Option.h"

class UnderlyingPath{
public:
  __host__ __device__  virtual double* GetPath()=0;
};

class MontecarloPath: public UnderlyingPath{
public:
  __host__ __device__  MontecarloPath(MarketData, double, int, StocasticProcess*, int EulerSubStep);
  __host__ __device__  ~MontecarloPath();
  __host__ __device__  double* GetPath();

private:
    double* _UnderlyingPath;
    StocasticProcess* _Process;
    MarketData _MarketInput;
    double _MaturityDate;
    int _NumberOfDatesToSimulate;
    int _EulerSubStep;
};

#endif
