#include <iostream>
#include <cmath>
#include "Gaussian.h"

 __host__ __device__ Gaussian::Gaussian(bool BoxMullerWithReExtraction){
  _BoxMullerWithReExtraction=BoxMullerWithReExtraction;
  _Status=true;
}
 __host__ __device__ double Gaussian::GetGaussianRandomNumber(double RN1, double RN2){
  if(_Status==true){
      double u=RN1;
      double v=RN2;
      if(_BoxMullerWithReExtraction==false){
          if(u==0) return this->GetGaussianRandomNumber(RN1, RN2);
          _SavedRandomNumber=sqrt(-2.*log(u))*sin(2*M_PI*v);
          _Status=false;
          return sqrt(-2.*log(u))*cos(2*M_PI*v);
      }
      else{
            u=2*u-1;
            v=2*v-1;
            double r=u*u+v*v;
            if(r==0 || r>=1) return this->GetGaussianRandomNumber(RN1, RN2);
            _SavedRandomNumber=v*sqrt(-2.*log(r)/r);
            _Status=false;
            return u*sqrt(-2.*log(r)/r);
      }
  }
  else{
    _Status=true;
    return _SavedRandomNumber;
  }
};
