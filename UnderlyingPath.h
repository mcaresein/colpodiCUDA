#ifndef _UnderlyingPath_h_
#define _UnderlyingPath_h_
#include "RandomGenerator.h"
#include "StocasticProcess.h"

class UnderlyingPath{
public:
//__host__ __device__ UnderlyingPath();
  __host__ __device__  virtual float* GetPath()=0;
};

class MontecarloPath: public UnderlyingPath{
public:
  __host__ __device__  MontecarloPath(float, float, float,RandomGenerator*, StocasticProcess*, int);
  __host__ __device__  ~MontecarloPath();
  __host__ __device__  float* GetPath();

private:
    float* _UnderlyingPath;
    RandomGenerator* _Generator;
    StocasticProcess* _Process;
    float _TInitial;
    float _SInitial;
    float _TFinal;
    int _NSteps;
};

#endif
