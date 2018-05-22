#include <iostream>
#include <cmath>
#include "RandomGenerator.h"
#include "DataTypes.h"

__host__ __device__  CombinedGenerator::CombinedGenerator(Seed S){
    _Sa=S.S1;
    _Sb=S.S2;
    _Sc=S.S3;
    _Sd=S.S4;
};

__host__ __device__  unsigned int CombinedGenerator::LCGStep(unsigned int seed, unsigned int a, unsigned long b){
	unsigned int x=(a*seed+b)%UINT_MAX;

	return x;
};

__host__ __device__  unsigned int CombinedGenerator::TausStep(unsigned int seed, unsigned int K1, unsigned int K2, unsigned int K3, unsigned long M){
	unsigned int b=(((seed<<K1)^seed)>>K2);
  unsigned int x=(((seed&M)<<K3)^b);

	return x;
};

__host__ __device__  double CombinedGenerator::Uniform(){
    _Sa=this->TausStep(_Sa, 13, 19, 12, 4294967294UL);
    _Sb=this->TausStep(_Sb, 2, 25, 4, 4294967288UL);
    _Sc=this->TausStep(_Sc, 3, 11, 17, 4294967280UL);
    _Sd=this->LCGStep(_Sd, 1664525, 1013904223UL);
    return 2.3283064365387e-10*(_Sa^_Sb^_Sc^_Sd);
};

__host__ __device__ double RandomGenerator::Gauss(){

        double u=this->Uniform();
        double v=this->Uniform();
        return sqrt(-2.*log(u))*cos(2*M_PI*v);

//## Stesso risultato ma calcolo ottimizzato per CPU. ##########################

/*      double u=2.*this->Uniform()-1;
        double v=2.*this->Uniform()-1;
        double r=u*u+v*v;
        if(r==0 || r>=1) return this->Gauss();
        return u*sqrt(-2.*log(r)/r); */

};
