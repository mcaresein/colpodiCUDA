#include <iostream>
#include <cmath>
#include "RandomGenerator.h"
#include "DataTypes.h"

__host__ __device__  CombinedGenerator::CombinedGenerator(Seed S, bool GPU){
    _Sa=S.S1;
    _Sb=S.S2;
    _Sc=S.S3;
    _Sd=S.S4;
    _GPU=GPU;
    _Status=true;
};

__host__ __device__  unsigned int CombinedGenerator::LCGStep(unsigned int &seed, unsigned int a, unsigned long b){
	return seed=(a*seed+b)%UINT_MAX;

};

__host__ __device__  unsigned int CombinedGenerator::TausStep(unsigned int &seed, unsigned int K1, unsigned int K2, unsigned int K3, unsigned long M){
	unsigned int b=(((seed<<K1)^seed)>>K2);
  return seed=(((seed&M)<<K3)^b);

};

__host__ __device__  double CombinedGenerator::GetUniformRandomNumber(){
    return 2.3283064365387e-10*(TausStep(_Sa, 13, 19, 12, 4294967294UL)^TausStep(_Sb, 2, 25, 4, 4294967288UL)^TausStep(_Sc, 3, 11, 17, 4294967280UL)^LCGStep(_Sd, 1664525, 1013904223UL));
};

__host__ __device__ double RandomGenerator::GetGaussianRandomNumber(){

        if(_Status==true){
              double u=this->GetUniformRandomNumber();
              double v=this->GetUniformRandomNumber();
              if(_GPU==true){
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
