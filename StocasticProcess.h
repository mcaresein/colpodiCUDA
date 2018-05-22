/*####################################################################################################
# Classi che implementano i metodi per simulare gli step temporali dell'evoluzione del prezzo del    #
# sottostante secondo processo esatto (ExactLogNormalProcess) o approssimato (EulerLogNormalProcess) #
####################################################################################################*/

#ifndef _StocasticProcess_h_
#define _StocasticProcess_h_

class StocasticProcess{
public:
    __host__ __device__  virtual double Step(double, double, double)=0;

};

class ExactLogNormalProcess: public StocasticProcess{
public:
  __host__ __device__  ExactLogNormalProcess(double, double);
  __host__ __device__  double Step(double ,double, double);
private:
  double _volatility, _drift;
};

class EulerLogNormalProcess: public StocasticProcess{
public:
  __host__ __device__  EulerLogNormalProcess(double, double);
  __host__ __device__  double Step(double, double, double);
private:
  double _volatility, _drift;
};

#endif
