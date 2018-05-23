/*##############################################################################################################################################################
# Pricer MonteCarlo di opzioni la cui dinamica e' determinata da processi lognormali esatti o approssimati.                                                    #
#                                                                                                                                                              #
# Usage: ./Pricer                                                                                                                                              #
# Speficicare: Dati di input del processo (MarketData), Dati di input dell'opzione (OptionData), tipo di Pay Off (guarda in PayOff.h per quelli implementati), #
#             tipo di processo (guarda in StocasticProcess.h per quelli implementati).                                                                         #
#                                                                                                                                                              #
# Output: Prezzo stimato secondo il Pay Off specificato e corrispondente errore MonteCarlo.                                                                    #
##############################################################################################################################################################*/

#include <iostream>
#include <cstdio>
#include <ctime>
#include "Pricer.h"
#include "Statistics.h"
#include "DataTypes.h"
#include "MainFunctions.cu"

#define STREAMS 5000
#define THREADS 5120

using namespace std;


int main(){

//## Inizializzazione parametri di mercato e opzione. ##########################

    MarketData MarketInput;
    MarketInput.Volatility=0.3;
    MarketInput.Drift=0.04;
    MarketInput.SInitial=100;

    OptionData OptionInput;
    OptionInput.TInitial=0;
    OptionInput.TFinal=1;
    OptionInput.NSteps=12;
    OptionInput.StrikePrice=100;

//## Allocazione di memoria. ###################################################

    //Allocation(THREADS, CPU);

    double *PayOffsGPU = new double[THREADS];
    double *PayOffs2GPU = new double[THREADS];
    double *PayOffsCPU = new double[THREADS];
    double *PayOffs2CPU = new double[THREADS];
    Seed *S= new Seed[THREADS];

    double *_PayOffsGPU;
    double *_PayOffs2GPU;
    Seed *_S;

    size_t sizeS = THREADS * sizeof(Seed);
    size_t sizePO = THREADS * sizeof(double);

    cudaMalloc((void**)& _PayOffsGPU, sizePO);
    cudaMalloc((void**)& _PayOffs2GPU, sizePO);
    cudaMalloc((void**)& _S, sizeS);

//## Costruzione vettore dei seed. #############################################

    srand(17*17);

    for(int i=0; i<THREADS; i++){
        S[i].S1=rand()%(UINT_MAX-128)+128;
        S[i].S2=rand()%(UINT_MAX-128)+128;
        S[i].S3=rand()%(UINT_MAX-128)+128;
        S[i].S4=rand();
    }

    cudaMemcpy(_S, S, sizeS, cudaMemcpyHostToDevice);

//## Calcolo dei PayOff su GPU. ################################################

    int blockSize=512;
    int gridSize = (THREADS + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    Kernel<<<gridSize, blockSize>>>(_S, _PayOffsGPU, _PayOffs2GPU, STREAMS, MarketInput, OptionInput);
    cudaEventRecord(stop);

    cudaMemcpy(PayOffsGPU, _PayOffsGPU, sizePO, cudaMemcpyDeviceToHost);
    cudaMemcpy(PayOffs2GPU, _PayOffs2GPU, sizePO, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

//## Calcolo dei PayOff su CPU. ################################################

    clock_t startcpu;
    double duration;

    startcpu = clock();
    KernelSimulator(S, PayOffsCPU, PayOffs2CPU, STREAMS, MarketInput, OptionInput, THREADS);
    duration = (clock() - startcpu ) / (double) CLOCKS_PER_SEC;

//## Calcolo PayOff ed errore monte carlo a partire dai valori di PayOff simulati. ##

    Statistics OptionGPU(PayOffsGPU, PayOffs2GPU, THREADS, STREAMS);
    Statistics OptionCPU(PayOffsCPU, PayOffs2CPU, THREADS, STREAMS);

//## Stampa a schermo dei valori. ##############################################

    cout<<"Valori GPU"<<endl;
    cout<<"Prezzo: "<<OptionGPU.GetPrice()<<endl;
    cout<<"Errore MonteCarlo: "<<OptionGPU.GetMCError()<<endl;
    cout<<"Tempo di calcolo: "<<milliseconds<<" ms"<<endl<<endl;

    cout<<"Valori CPU"<<endl;
    cout<<"Prezzo: "<<OptionCPU.GetPrice()<<endl;
    cout<<"Errore MonteCarlo: "<<OptionCPU.GetMCError()<<endl;
    cout<<"Tempo di calcolo: "<<duration*1000<<" ms"<<endl;


//## Liberazione memoria. ######################################################

    //DeAllocation();
    cudaFree(_PayOffsGPU);
    cudaFree(_PayOffs2GPU);
    cudaFree(_S);

    delete[] PayOffsGPU;
    delete[] PayOffsCPU;
    delete[] PayOffs2GPU;
    delete[] PayOffs2CPU;
    delete[] S;

    return 0;
}
