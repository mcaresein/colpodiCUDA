#include <iostream>
#include <cmath>
#include "randomgen.h"

__host__ __device__ Randomgen::Randomgen(int Sa, int Sb, int Sc, int Sd){
    _Sa=Sa;
    _Sb=Sb;
    _Sc=Sc;
    _Sd=Sd;
};

__host__ __device__ unsigned int Randomgen::LCGStep(unsigned int seed, unsigned int a, unsigned long b){
	unsigned int x=(a*seed+b)%UINT_MAX;
//    double y=x;
//	double r=y/UINT_MAX;
	return x;
}

__host__ __device__ unsigned int Randomgen::TausStep(unsigned int seed, unsigned int K1, unsigned int K2, unsigned int K3, unsigned long M){
	unsigned int b=(((seed<<K1)^seed)>>K2);
    unsigned int x=(((seed&M)<<K3)^b);
//    double y=x;
//	double r=y/UINT_MAX;
	return x;
};

__host__ __device__ double Randomgen::Rand(){
    _Sa=this->TausStep(_Sa, 13, 19, 12, 4294967294UL);
    _Sb=this->TausStep(_Sb, 2, 25, 4, 4294967288UL);
    _Sc=this->TausStep(_Sc, 3, 11, 17, 4294967280UL);
    _Sd=this->LCGStep(_Sd, 1664525, 1013904223UL);
    return 2.3283064365387e-10*(_Sa^_Sb^_Sc^_Sd);
};

__host__ __device__ double Randomgen::Gauss(){
    #ifdef __CUDA_ARCH__
        double u=this->Rand();
        double v=this->Rand();
        return sqrt(-2.*log(u))*cos(2*M_PI*v);
    #else
        double u=2.*this->Rand()-1;
        double v=2.*this->Rand()-1;
        double r=u*u+v*v;
        if(r==0 || r>=1) return this->Gauss();
        return u*sqrt(-2.*log(r)/r);
    #endif
}
