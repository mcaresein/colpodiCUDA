#include <iostream>
#include <cmath>
#include "RandomGenerator.h"
//#include "RandomGeneratorCombined.h"
#include "Seed.h"

__host__ __device__ double RandomGenerator::GetGaussianRandomNumber(){

        if(_Status==true){
              double u=this->GetUniformRandomNumber();
              double v=this->GetUniformRandomNumber();
              if(_ReExtractionBoxMuller==false){
                    _SavedRandomNumber=sqrt(-2.*log(u))*sin(2*M_PI*v);
                    _Status=false;
                    return sqrt(-2.*log(u))*cos(2*M_PI*v);
              }
              else{
                    u=2*u-1;
                    v=2*v-1;
                    double r=u*u+v*v;
                    if(r==0 || r>=1) return this->GetGaussianRandomNumber();
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
#
