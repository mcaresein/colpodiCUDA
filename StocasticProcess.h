/*####################################################################################################
# Classi che implementano i metodi per simulare gli step temporali dell'evoluzione del prezzo del    #
# sottostante secondo processo esatto (ExactLogNormalProcess) o approssimato (EulerLogNormalProcess) #
####################################################################################################*/

#ifndef _StocasticProcess_h_
#define _StocasticProcess_h_

class StocasticProcess{
public:
    __host__ __device__  virtual double Step(double InitialPrice, double TimeStep, double RandomNumber)=0;

};

class ExactLogNormalProcess: public StocasticProcess{
public:
  __host__ __device__  ExactLogNormalProcess(double Volatility, double Drift);
  __host__ __device__  double Step(double InitialPrice, double TimeStep, double RandomNumber);
private:
  double _Volatility, _Drift;
};

class EulerLogNormalProcess: public StocasticProcess{
public:
  __host__ __device__  EulerLogNormalProcess(double Volatility, double Drift);
  __host__ __device__  double Step(double InitialPrice, double TimeStep, double RandomNumber);
private:
  double _Volatility, _Drift;
};

#endif
