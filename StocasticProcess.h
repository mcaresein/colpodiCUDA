#ifndef _StocasticProcess_h_
#define _StocasticProcess_h_

class StocasticProcess{
public:
//  __host__ __device__ StocasticProcess();
  __host__ __device__  virtual float Step(float S, float T, float w)=0;      //sottinteso step esatto (quando esiste)

};

class ExactLogNormalProcess: public StocasticProcess{
public:
  __host__ __device__  ExactLogNormalProcess(float volatility, float drift);
  __host__ __device__  float Step(float S, float T, float w);
private:
  float _volatility, _drift;
};

class EulerLogNormalProcess: public StocasticProcess{
public:
  __host__ __device__  EulerLogNormalProcess(float volatility, float drift);
  __host__ __device__  float Step(float S, float T, float w);
private:
  float _volatility, _drift;
};

#endif
