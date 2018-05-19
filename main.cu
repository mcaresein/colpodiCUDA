/*##############################################################################################################################################################
# Pricer MonteCarlo di opzioni la cui dinamica e' determinata da processi lognormali esatti o approssimati.                                                    #
#                                                                                                                                                              #
# Usage: ./Pricer                                                                                                                                              #
# Speficicare: Dati di input del processo (MarketData), Dati di input dell'opzione (OptionData), tipo di Pay Off (guarda in PayOff.h per quelli implementati), #
#             tipo di processo (guarda in StocasticProcess.h per quelli implementati) e se fare i calcoli su CPU o GPU.                                        #
#                                                                                                                                                              #
# Output: Prezzo stimato secondo il Pay Off specificato e corrispondente errore MonteCarlo.                                                                    #
##############################################################################################################################################################*/

#include <iostream>
#include "Pricer.h"
#include "Statistics.h"
#include "DataTypes.h"

using namespace std;

//## Funzione che gira su GPU che restituisce due vettori con sommme dei PayOff e dei PayOff quadrati. ##

__global__ void Kernel(Seed* S, float* PayOffs, float* PayOffs2, int streams){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    MarketData MarketInput;
    MarketInput.Volatility=0.3;
    MarketInput.Drift=0.04;
    MarketInput.SInitial=100;

    OptionData OptionInput;
    OptionInput.TInitial=0;
    OptionInput.TFinal=1;
    OptionInput.NSteps=12;
    OptionInput.StrikePrice=100;

    MonteCarloPricer P(MarketInput, OptionInput, streams, S[i]);
    P.GetPrice();

    float SumPayOff=P.GetPayOff();
    float SumPayOff2=P.GetPayOff2();

    PayOffs[i]=SumPayOff;
    PayOffs2[i]=SumPayOff2;
}

//## Funzione che gira su CPU che restituisce due vettori con sommme dei PayOff e dei PayOff quadrati. ##

__host__ void KernelSimulator(Seed* S, float* PayOffs, float* PayOffs2, int streams, int threads){

    MarketData MarketInput;
    MarketInput.Volatility=0.3;
    MarketInput.Drift=0.04;
    MarketInput.SInitial=100;

    OptionData OptionInput;
    OptionInput.TInitial=0;
    OptionInput.TFinal=1;
    OptionInput.NSteps=12;
    OptionInput.StrikePrice=100;

    for(int i=0; i<threads; i++){
        MonteCarloPricer P(MarketInput, OptionInput, streams, S[i]);
        P.GetPrice();
        float SumPayOff=P.GetPayOff();
        float SumPayOff2=P.GetPayOff2();

        PayOffs[i]=SumPayOff;
        PayOffs2[i]=SumPayOff2;
    }
}

int main(){

    int streams=100;
    int threads=1024;

//## Costruzione vettore dei seed. #############################################

    srand(657);
    Seed *S= new Seed[threads];
    float *PayOffs = new float[threads];
    float *PayOffs2 = new float[threads];

    for(int i=0; i<threads; i++){
		S[i].S1=rand()+128;
        S[i].S2=rand()+128;
        S[i].S3=rand()+128;
        S[i].S4=rand()+128;
	}

    float *_PayOffs;
    float *_PayOffs2;
    Seed *_S;

//## Estrazione vettori dei PayOff su GPU. #####################################

    size_t sizeS = threads * sizeof(Seed);
    size_t sizePO = threads * sizeof(float);

    cudaMalloc((void**)& _PayOffs, sizePO);
    cudaMalloc((void**)& _PayOffs2, sizePO);
    cudaMalloc((void**)& _S, sizeS);

    cudaMemcpy(_S, S, sizeS, cudaMemcpyHostToDevice);

    int blockSize=512;
    int gridSize = (threads + blockSize - 1) / blockSize;

    Kernel<<<gridSize, blockSize>>>(_S, _PayOffs, _PayOffs2, streams);

    cudaMemcpy(PayOffs, _PayOffs, sizePO, cudaMemcpyDeviceToHost);
    cudaMemcpy(PayOffs2, _PayOffs2, sizePO, cudaMemcpyDeviceToHost);

//## Estrazione vettori dei PayOff su CPU. #####################################

//    KernelSimulator(S, PayOffs, PayOffs2, streams, threads);

//## Calcolo PayOff mediato ed errore monte carlo a partire dai vettori di PayOff estratti. ##

    Statistics Option(PayOffs, PayOffs2, threads, streams);

//## Stampa a schermo dei valori. ##############################################

    cout<<"Prezzo: "<<Option.GetPrice()<<endl;
    cout<<"Errore MonteCarlo: "<<Option.GetMCError()<<endl;

//## Liberazione memoria. ######################################################

    cudaFree(_PayOffs);
    cudaFree(_PayOffs2);
    cudaFree(_S);

    delete[] PayOffs;
    delete[] PayOffs2;
    delete[] S;

    return 0;
}
