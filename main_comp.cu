/*##############################################################################################################################################################
# Pricer MonteCarlo di opzioni la cui dinamica e' determinata da processi lognormali esatti o approssimati.                                                    #
# Confronto prestazioni CPU e GPU                                                                                                                              #
# Usage: ./pricer_comp                                                                                                                                         #
# Speficicare: Dati di input del processo (MarketData), Dati di input dell'opzione (OptionData), tipo di Opzione (guarda in Option.h per quelle implementate), #
#             tipo di processo (guarda in StocasticProcess.h per quelli implementati).                                                                         #
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
    OptionDataContainer OptionInput;
    SimulationParameters Parameters;
    GPUData GPUInput;

    Reader(MarketInput, OptionInput, GPUInput, Parameters);

//## Allocazione di memoria. ###################################################

    Statistics* PayOffsGPU;
    Statistics* PayOffsCPU;
    Seed *SeedVector;

    Statistics* _PayOffsGPU;
    Seed *_SeedVector;

    size_t sizeSeedVector = GPUInput.Threads * sizeof(Seed);
    size_t sizeDevStVector = GPUInput.Threads * sizeof(Statistics);

    MemoryAllocationGPUandCPU(& PayOffsGPU, & PayOffsCPU,  & SeedVector, & _PayOffsGPU, & _SeedVector, sizeSeedVector, sizeDevStVector, GPUInput.Threads);

//## Costruzione vettore dei seed. #############################################

    GetSeeds(SeedVector, GPUInput.Threads);

    cudaMemcpy(_SeedVector, SeedVector, sizeSeedVector, cudaMemcpyHostToDevice);

//## Calcolo dei PayOff su GPU. ################################################

    cout<<"Simulazione GPU..."<<endl;

    int gridSize = (GPUInput.Threads + GPUInput.BlockSize - 1) / GPUInput.BlockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    Kernel<<<gridSize, GPUInput.BlockSize>>>(_SeedVector, _PayOffsGPU, GPUInput.Streams, MarketInput, OptionInput, Parameters);
    cudaEventRecord(stop);

    cudaMemcpy(PayOffsGPU, _PayOffsGPU, sizeDevStVector, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

//## Calcolo dei PayOff su CPU. ################################################

    cout<<"Simulazione CPU..."<<endl;

    double duration;
    clock_t startcpu;


    startcpu = clock();
    KernelSimulator(SeedVector, PayOffsCPU, GPUInput.Streams, MarketInput, OptionInput, Parameters, GPUInput.Threads);
    duration = (clock() - startcpu ) / (double) CLOCKS_PER_SEC;

    Statistics FinalStatisticsGPU;

    for (int i=0; i<GPUInput.Threads; i++){
      FinalStatisticsGPU=FinalStatisticsGPU+PayOffsGPU[i];
    };

    Statistics FinalStatisticsCPU;


    for (int i=0; i<GPUInput.Threads; i++){
      FinalStatisticsCPU=FinalStatisticsCPU+PayOffsCPU[i];
    };

//## Calcolo e stampa su file dei valori. ######################################

    cout<<endl<<"Valori GPU:"<<endl;
    FinalStatisticsGPU.Print(OptionInput.MaturityDate, MarketInput.Drift);
    cout<<"Tempo di calcolo: "<<milliseconds<<" ms"<<endl<<endl;
    cout<<"Valori CPU:"<<endl;
    FinalStatisticsCPU.Print(OptionInput.MaturityDate, MarketInput.Drift);
    cout<<"Tempo di calcolo: "<<duration*1000<<" ms"<<endl<<endl;

    FinalStatisticsGPU.Print(OptionInput.MaturityDate, MarketInput.Drift,"DATA/outputGPU.dat");
    FinalStatisticsCPU.Print(OptionInput.MaturityDate, MarketInput.Drift,"DATA/outputCPU.dat");

//## Controllo #################################################################

    double GPUPrice=FinalStatisticsGPU.GetMean();
    double CPUPrice=FinalStatisticsCPU.GetMean();
    double GPUError=FinalStatisticsGPU.GetStDev();
    double CPUError=FinalStatisticsCPU.GetStDev();

    if(GPUPrice==CPUPrice)
        cout<<"I prezzi coincidono!"<<endl;

    else{
        cout<<"I prezzi NON coincidono!"<<endl;
        cout<<"Discrepanza: "<<GPUPrice-CPUPrice<<endl;
    }
    if(GPUError==CPUError)
        cout<<"Gli errori coincidono!"<<endl;
    else{
        cout<<"Gli errori NON coincidono!"<<endl;
        cout<<"Discrepanza: "<<GPUError-CPUError<<endl;
    }

//## Liberazione memoria. ######################################################

    MemoryDeallocationGPUandCPU(PayOffsGPU, PayOffsCPU, SeedVector, _PayOffsGPU, _SeedVector);

    return 0;
}
