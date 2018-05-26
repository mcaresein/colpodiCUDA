#ifndef _MainFunctions_cu_
#define _MainFunctions_cu_

#include <iostream>
#include "MonteCarloPricer.h"
#include "Statistics.h"
#include "RandomGenerator.h"
#include "DataTypes.h"
#include "Option.h"

/*############################ Kernel Function ############################*/

__host__ __device__ void TrueKernel(Seed* SeedVector, double* PayOffs, double* PayOffs2, int streams, MarketData MarketInput, OptionData OptionInput, int cont){

  RandomGenerator* Generator= new CombinedGenerator(SeedVector[cont], true);
  Option* Option=new OptionPlainVanillaCall(OptionInput);
  StocasticProcess* Process=new ExactLogNormalProcess(MarketInput.Volatility, MarketInput.Drift);


  MonteCarloPricer P(MarketInput, Option, Generator, Process, streams);
  P.ComputePrice();

  double SumPayOff=P.GetPayOff();
  double SumPayOff2=P.GetPayOff2();

  PayOffs[cont]=SumPayOff;
  PayOffs2[cont]=SumPayOff2;

}

__global__ void Kernel(Seed* SeedVector, double* PayOffs, double* PayOffs2, int streams, MarketData MarketInput, OptionData OptionInput){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    TrueKernel(SeedVector, PayOffs, PayOffs2, streams, MarketInput, OptionInput, i);
};

//## Funzione che gira su CPU che restituisce due vettori con sommme dei PayOff e dei PayOff quadrati. ##

__host__ void KernelSimulator(Seed* SeedVector, double* PayOffs, double* PayOffs2, int streams, MarketData MarketInput, OptionData OptionInput, int threads){

    for(int i=0; i<threads; i++) TrueKernel(SeedVector, PayOffs, PayOffs2, streams, MarketInput, OptionInput, i);

};

/*############################ Main Function ############################

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

  void DeAllocation(){
    cudaFree(_PayOffsGPU);
    cudaFree(_PayOffs2GPU);
    cudaFree(_S);

    delete[] PayOffsGPU;
    delete[] PayOffsCPU;
    delete[] PayOffs2GPU;
    delete[] PayOffs2CPU;
    delete[] S;
};
*/
#endif
