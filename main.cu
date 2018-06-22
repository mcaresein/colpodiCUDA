/*##############################################################################################################################################################
# Pricer MonteCarlo di opzioni la cui dinamica e' determinata da processi lognormali esatti o approssimati.                                                    #
#                                                                                                                                                              #
# Usage: ./pricer                                                                                                                                              #
# Speficicare: Dati di input del processo (MarketData), Dati di input dell'opzione (OptionData), tipo di opzione (guarda in Option.h per quelle implementate), #
#             tipo di processo (guarda in StocasticProcess.h per quelli implementati). Vedi file input.conf in DATA                                            #
#                                                                                                                                                              #
# Output: Prezzo stimato secondo il Pay Off specificato e corrispondente errore MonteCarlo.                                                                    #
##############################################################################################################################################################*/

#include <iostream>
#include <cstdio>
#include <ctime>
#include "MonteCarloPricer.h"
#include "Statistics.h"
#include "Seed.h"
#include "MarketData.h"
#include "OptionData.h"
#include "GPUData.h"
#include "SimulationParameters.h"
#include "KernelFunctions.cu"
#include "Utilities.cu"

using namespace std;

int main(){
//## Inizializzazione parametri di mercato e opzione. ##########################

    MarketData MarketInput;
    OptionData OptionInput;
    SimulationParameters Parameters;
    GPUData GPUInput;

    Reader(MarketInput, OptionInput, GPUInput, Parameters);

//## Allocazione di memoria. ###################################################

    Statistics* PayOffs;
    Seed *SeedVector;

    Statistics* _PayOffs;
    Seed *_SeedVector;

    size_t sizeSeedVector = GPUInput.Threads * sizeof(Seed);
    size_t sizeDevStVector = GPUInput.Threads * sizeof(Statistics);

    MemoryAllocationGPU(& PayOffs,  & SeedVector, & _PayOffs, & _SeedVector, sizeSeedVector, sizeDevStVector, GPUInput.Threads);

//## Costruzione vettore dei seed. #############################################

    GetSeeds(SeedVector, GPUInput.Threads);

    cudaMemcpy(_SeedVector, SeedVector, sizeSeedVector, cudaMemcpyHostToDevice);

//## Calcolo dei PayOff su GPU. ################################################

    cout<<"Simulazione..."<<endl;

    int gridSize = (GPUInput.Threads + GPUInput.BlockSize - 1) / GPUInput.BlockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    Kernel<<<gridSize, GPUInput.BlockSize>>>(_SeedVector, _PayOffs, GPUInput.Streams, MarketInput, OptionInput, Parameters);
    cudaEventRecord(stop);

    cudaMemcpy(PayOffs, _PayOffs, sizeDevStVector, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

//## Calcolo e stampa su file dei valori. ######################################

    Statistics FinalStatistics;

    cout<<endl<<"Valori ottenuti: "<<endl;
    FinalStatistics.Print(PayOffs, GPUInput.Threads);
    cout<<"Tempo di calcolo: "<<milliseconds<<" ms"<<endl<<endl;
    FinalStatistics.Print("DATA/output.dat", PayOffs, GPUInput.Threads);


//## Liberazione memoria. ######################################################

    MemoryDeallocationGPU(PayOffs, SeedVector, _PayOffs, _SeedVector);

    cudaDeviceReset();

    return 0;
}
