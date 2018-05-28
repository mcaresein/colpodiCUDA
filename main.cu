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
#include "MonteCarloPricer.h"
#include "Statistics.h"
#include "DataTypes.h"
#include "KernelFunctions.cu"
#include "Utilities.cu"

using namespace std;

int main(){

//## Inizializzazione parametri di mercato e opzione. ##########################

    int THREADS;
    int STREAMS;

    MarketData MarketInput;
    OptionData OptionInput;

    Reader(MarketInput, OptionInput, THREADS, STREAMS);

//## Allocazione di memoria. ###################################################

    double *PayOffsGPU = new double[THREADS];
    double *PayOffs2GPU = new double[THREADS];
    double *PayOffsCPU = new double[THREADS];
    double *PayOffs2CPU = new double[THREADS];
    Seed *SeedVector= new Seed[THREADS];

    double *_PayOffsGPU;
    double *_PayOffs2GPU;
    Seed *_SeedVector;

    size_t sizeSeedVector = THREADS * sizeof(Seed);
    size_t sizePO = THREADS * sizeof(double);

    cudaMalloc((void**)& _PayOffsGPU, sizePO);
    cudaMalloc((void**)& _PayOffs2GPU, sizePO);
    cudaMalloc((void**)& _SeedVector, sizeSeedVector);

//## Costruzione vettore dei seed. #############################################

    GetSeeds(SeedVector, THREADS);

    cudaMemcpy(_SeedVector, SeedVector, sizeSeedVector, cudaMemcpyHostToDevice);

//## Calcolo dei PayOff su GPU. ################################################

    int blockSize=512;
    int gridSize = (THREADS + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    Kernel<<<gridSize, blockSize>>>(_SeedVector, _PayOffsGPU, _PayOffs2GPU, STREAMS, MarketInput, OptionInput);
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
    KernelSimulator(SeedVector, PayOffsCPU, PayOffs2CPU, STREAMS, MarketInput, OptionInput, THREADS);
    duration = (clock() - startcpu ) / (double) CLOCKS_PER_SEC;

//## Calcolo PayOff ed errore monte carlo a partire dai valori di PayOff simulati. ##

    Statistics OptionGPU(PayOffsGPU, PayOffs2GPU, THREADS, STREAMS);
    Statistics OptionCPU(PayOffsCPU, PayOffs2CPU, THREADS, STREAMS);

//## Stampa su file dei valori. ##############################################

    cout<<"Valori GPU"<<endl;
    OptionGPU.Print();
    OptionGPU.Print("DATA/outputGPU.dat");
    cout<<"Tempo di calcolo: "<<milliseconds<<" ms"<<endl<<endl;

    cout<<"Valori CPU"<<endl;
    OptionCPU.Print();
    OptionCPU.Print("DATA/outputCPU.dat");
    cout<<"Tempo di calcolo: "<<duration*1000<<" ms"<<endl;


//## Liberazione memoria. ######################################################

    cudaFree(_PayOffsGPU);
    cudaFree(_PayOffs2GPU);
    cudaFree(_SeedVector);

    delete[] PayOffsGPU;
    delete[] PayOffsCPU;
    delete[] PayOffs2GPU;
    delete[] PayOffs2CPU;
    delete[] SeedVector;

    return 0;
}
