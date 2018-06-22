#include <iostream>
#include <fstream>
#include <string>
#include "Seed.h"
#include "MarketData.h"
#include "OptionData.h"
#include "GPUData.h"
#include "SimulationParameters.h"

using namespace std;

void Reader(MarketData &MarketInput, OptionData &OptionInput, GPUData &GPUInput, SimulationParameters &Parameters){

    string InputFile="DATA/input.conf";
    cout<<"Lettura file di input: "<<InputFile<<" ..."<<endl;
    fstream file;
    file.open(InputFile.c_str() , ios::in);

    if(file.fail()){
        cout<< "ERRORE: file di configurazione non trovato. "<<  endl;
    }
    string temp, word;
    int Threads=0, Streams=0, BlockSize=0;
    int EulerApproximation=0, OptionType=0;
    double Volatility=0, Drift=0;
    double InitialPrice=0, MaturityDate=0, StrikePrice=0;
    int DatesToSimulate=0, EulerSubStep=1;
    double ParamK=0, ParamB=0, ParamN=0;

    while (!file.eof()){
        file>>word;
        if (word=="THREADS") {
            file>> temp;
            Threads=atoi(temp.c_str());
        }
        if (word=="STREAMS"){
            file>> temp;
            Streams=atoi(temp.c_str());
        }
        if (word=="BLOCK_SIZE"){
            file>> temp;
            BlockSize=atoi(temp.c_str());
        }
        if (word=="EULER_APPROX"){
            file>> temp;
            EulerApproximation=atoi(temp.c_str());
        }
        if (word=="OPTION_TYPE"){
            file>> temp;
            OptionType=atoi(temp.c_str());
        }
        if (word=="VOLATILITY"){
            file>> temp;
            Volatility=atof(temp.c_str());
        }
        if (word=="DRIFT"){
            file>> temp;
            Drift=atof(temp.c_str());
        }
        if (word=="INITIAL_PRICE"){
            file>> temp;
            InitialPrice=atof(temp.c_str());
        }
        if (word=="MATURITY_DATE"){
            file>> temp;
            MaturityDate=atof(temp.c_str());
        }
        if (word=="DATES_TO_SIMULATE"){
            file>> temp;
            DatesToSimulate=atoi(temp.c_str());
        }
        if (word=="STRIKE_PRICE"){
            file>> temp;
            StrikePrice=atof(temp.c_str());
        }
        if (word=="PARAMETER_K"){
            file>> temp;
            ParamK=atof(temp.c_str());
        }
        if (word=="PARAMETER_N"){
            file>> temp;
            ParamN=atof(temp.c_str());
        }
        if (word=="PARAMETER_B"){
            file>> temp;
            ParamB=atof(temp.c_str());
        }
        if (word=="EULER_SUB_STEP"){
            file>> temp;
            EulerSubStep=atof(temp.c_str());
        }
    }

    file.close();

    GPUInput.Threads=Threads;
    GPUInput.Streams=Streams;
    GPUInput.BlockSize=BlockSize;

    bool EulerBool=EulerApproximation;
    Parameters.EulerApprox=EulerBool;
    Parameters.OptionType=OptionType;

    MarketInput.Volatility=Volatility;
    MarketInput.Drift=Drift;
    MarketInput.EquityInitialPrice=InitialPrice;

    if(EulerBool==false)
        EulerSubStep=1;

    OptionInput.MaturityDate=MaturityDate;
    OptionInput.NumberOfDatesToSimulate=DatesToSimulate,
    OptionInput.EulerSubStep=EulerSubStep;
    OptionInput.StrikePrice=StrikePrice;
    OptionInput.OptionTypeCallOrPut=OptionType;
    OptionInput.B=ParamB;
    OptionInput.N=ParamN;
    OptionInput.K=ParamK;

    /*
    if(OptionType==0){
        *OptionInput = new OptionDataForward(MaturityDate, DatesToSimulate, EulerSubStep);
    }
    if(OptionType==1 || OptionType==2){
        *OptionInput = new OptionDataPlainVanilla(MaturityDate, DatesToSimulate, EulerSubStep, StrikePrice, OptionType);
    }
    if(OptionType==3){
        *OptionInput = new OptionDataAbsolutePerformanceBarrier(MaturityDate, DatesToSimulate, EulerSubStep, ParamB, ParamN, ParamK);
    }
    */
}

void GetSeeds(Seed* SeedVector, int THREADS){

    srand(17*17);

    for(int i=0; i<THREADS; i++){
        SeedVector[i].S1=rand()%(UINT_MAX-128)+128;
        SeedVector[i].S2=rand()%(UINT_MAX-128)+128;
        SeedVector[i].S3=rand()%(UINT_MAX-128)+128;
        SeedVector[i].S4=rand();
    }
};

void MemoryAllocationGPUandCPU(Statistics** PayOffsGPU, Statistics** PayOffsCPU, Seed** SeedVector, Statistics** _PayOffsGPU, Seed** _SeedVector, size_t sizeSeedVector, size_t sizeDevStVector, int THREADS){
    cout<<"Allocazione della memoria..."<<endl;
    *PayOffsGPU=new Statistics[THREADS];
    *PayOffsCPU = new Statistics[THREADS];
    *SeedVector= new Seed[THREADS];

    cudaMalloc((void**)& *_PayOffsGPU, sizeDevStVector);
    cudaMalloc((void**)& *_SeedVector, sizeSeedVector);
};

void MemoryAllocationGPU(Statistics** PayOffsGPU, Seed** SeedVector, Statistics** _PayOffsGPU, Seed** _SeedVector, size_t sizeSeedVector, size_t sizeDevStVector, int THREADS){
    cout<<"Allocazione della memoria..."<<endl;
    *PayOffsGPU=new Statistics[THREADS];
    *SeedVector= new Seed[THREADS];

    cudaMalloc((void**)& *_PayOffsGPU, sizeDevStVector);
    cudaMalloc((void**)& *_SeedVector, sizeSeedVector);
};

void MemoryDeallocationGPUandCPU(Statistics* PayOffsGPU, Statistics* PayOffsCPU, Seed* SeedVector, Statistics* _PayOffsGPU, Seed* _SeedVector){
    delete[] PayOffsGPU;
    delete[] PayOffsCPU;
    delete[] SeedVector;

    cudaFree(_PayOffsGPU);
    cudaFree(_SeedVector);

    cudaDeviceReset();
};

void MemoryDeallocationGPU(Statistics* PayOffsGPU, Seed* SeedVector, Statistics* _PayOffsGPU, Seed* _SeedVector){
    delete[] PayOffsGPU;
    delete[] SeedVector;

    cudaFree(_PayOffsGPU);
    cudaFree(_SeedVector);

    cudaDeviceReset();
};
