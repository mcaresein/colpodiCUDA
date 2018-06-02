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

    DevStatistics* PayOffsGPU;
    DevStatistics* PayOffsCPU;
    Seed *SeedVector;

    DevStatistics* _PayOffsGPU;
    Seed *_SeedVector;

    size_t sizeSeedVector = THREADS * sizeof(Seed);
    size_t sizeDevStVector = THREADS * sizeof(DevStatistics);

    MemoryAllocation(& PayOffsGPU, & PayOffsCPU,  & SeedVector, & _PayOffsGPU, & _SeedVector, sizeSeedVector, sizeDevStVector, THREADS);

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
    Kernel<<<gridSize, blockSize>>>(_SeedVector, _PayOffsGPU, STREAMS, MarketInput, OptionInput);
    cudaEventRecord(stop);

    cudaMemcpy(PayOffsGPU, _PayOffsGPU, sizeDevStVector, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

//## Calcolo dei PayOff su CPU. ################################################

    clock_t startcpu;
    double duration;

    startcpu = clock();
    KernelSimulator(SeedVector, PayOffsCPU, STREAMS, MarketInput, OptionInput, THREADS);
    duration = (clock() - startcpu ) / (double) CLOCKS_PER_SEC;

//## Calcolo PayOff ed errore monte carlo a partire dai valori di PayOff simulati. ##

    HostStatistics OptionGPU(PayOffsGPU, THREADS, STREAMS);
    HostStatistics OptionCPU(PayOffsCPU, THREADS, STREAMS);

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

    MemoryDeallocation(PayOffsGPU, PayOffsCPU, SeedVector, _PayOffsGPU, _SeedVector);

    return 0;
}
