#ifndef _MainFunctions_cu_
#define _MainFunctions_cu_

#include <iostream>
#include "Pricer.h"
#include "Statistics.h"
#include "DataTypes.h"

/*############################ Kernel Function ############################*/


__global__ void Kernel(Seed* S, double* PayOffs, double* PayOffs2, int streams, MarketData MarketInput, OptionData OptionInput){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    MonteCarloPricer P(MarketInput, OptionInput, streams, S[i]);
    P.ComputePrice();

    double SumPayOff=P.GetPayOff();
    double SumPayOff2=P.GetPayOff2();

    PayOffs[i]=SumPayOff;
    PayOffs2[i]=SumPayOff2;
};

//## Funzione che gira su CPU che restituisce due vettori con sommme dei PayOff e dei PayOff quadrati. ##

__host__ void KernelSimulator(Seed* S, double* PayOffs, double* PayOffs2, int streams, MarketData MarketInput, OptionData OptionInput, int threads){

    for(int i=0; i<threads; i++){
        MonteCarloPricer P(MarketInput, OptionInput, streams, S[i]);
        P.ComputePrice();

        double SumPayOff=P.GetPayOff();
        double SumPayOff2=P.GetPayOff2();

        PayOffs[i]=SumPayOff;
        PayOffs2[i]=SumPayOff2;
    }
};

/*############################ Main Function ############################*/


void Allocation(unsigned int THREADS, bool cpu){
    double *PayOffsGPU = new double[THREADS];
    double *PayOffs2GPU = new double[THREADS];
    if (cpu) double *PayOffsCPU = new double[THREADS];
    if (cpu) double *PayOffs2CPU = new double[THREADS];
    Seed *S= new Seed[THREADS];

    double *_PayOffsGPU;
    double *_PayOffs2GPU;
    Seed *_S;

    size_t sizeS = THREADS * sizeof(Seed);
    size_t sizePO = THREADS * sizeof(double);

    cudaMalloc((void**)& _PayOffsGPU, sizePO);
    cudaMalloc((void**)& _PayOffs2GPU, sizePO);
    cudaMalloc((void**)& _S, sizeS);
};

  void ~Allocation(){
    cudaFree(_PayOffsGPU);
    cudaFree(_PayOffs2GPU);
    cudaFree(_S);

    delete[] PayOffsGPU;
    delete[] PayOffsCPU;
    delete[] PayOffs2GPU;
    delete[] PayOffs2CPU;
    delete[] S;
};

#endif
