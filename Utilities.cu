#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void Reader(MarketData &MarketInput, OptionData &OptionInput, int &THREADS, int &STREAMS, SimulationParameters &Parameters){
    fstream file("DATA/input.conf");
    string line;
    double data[14];
    int i=0;
    while (getline(file, line)){
        if (line[0] == '#') continue;
        data[i]=atof(line.c_str());
        i++;
    }
    file.close();

    THREADS=data[0];
    STREAMS=data[1];
    bool EulerApproximation=data[2];
    Parameters.EulerApprox=EulerApproximation;
    Parameters.OptionType=data[3];

    MarketInput.Volatility=data[4];
    MarketInput.Drift=data[5];
    MarketInput.EquityInitialPrice=data[6];

    OptionInput.TInitial=data[7];
    OptionInput.MaturityDate=data[8];
    OptionInput.NumberOfDatesToSimulate=data[9];
    OptionInput.StrikePrice=data[10];
    OptionInput.K=data[11];
    OptionInput.N=data[12];
    OptionInput.B=data[13];
};

void GetSeeds(Seed* SeedVector, int THREADS){

    srand(17*17);

    for(int i=0; i<THREADS; i++){
        SeedVector[i].S1=rand()%(UINT_MAX-128)+128;
        SeedVector[i].S2=rand()%(UINT_MAX-128)+128;
        SeedVector[i].S3=rand()%(UINT_MAX-128)+128;
        SeedVector[i].S4=rand();
    }
};

void MemoryAllocation(DevStatistics** PayOffsGPU, DevStatistics** PayOffsCPU, Seed** SeedVector, DevStatistics** _PayOffsGPU, Seed** _SeedVector, size_t sizeSeedVector, size_t sizeDevStVector, int THREADS){
    *PayOffsGPU=new DevStatistics[THREADS];
    *PayOffsCPU = new DevStatistics[THREADS];
    *SeedVector= new Seed[THREADS];

    cudaMalloc((void**)& *_PayOffsGPU, sizeDevStVector);
    cudaMalloc((void**)& *_SeedVector, sizeSeedVector);
};

void MemoryDeallocation(DevStatistics* PayOffsGPU, DevStatistics* PayOffsCPU, Seed* SeedVector, DevStatistics* _PayOffsGPU, Seed* _SeedVector){
    delete[] PayOffsGPU;
    delete[] PayOffsCPU;
    delete[] SeedVector;

    cudaFree(_PayOffsGPU);
    cudaFree(_SeedVector);
};
