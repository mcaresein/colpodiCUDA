#include "Statistics.h"
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

__host__ double Statistics::GetPrice(Statistics* PayOffs, int DimThreads){
	double SumP=0;
	int DimStreams=PayOffs[0].GetCont(); //vanno bene tutti gli elementi del vettore PayOff per estrarre _Cont??
	int N=DimThreads*DimStreams;

	for(int i=0; i<DimThreads; i++){
		SumP += PayOffs[i].GetPayOffs();
	}
	return SumP/N;
};

__host__ double Statistics::GetMCError(Statistics* PayOffs, int DimThreads){
	double SumP=0, SumP2=0;
	int DimStreams=PayOffs[0].GetCont();
	int N=DimThreads*DimStreams;

	for(int i=0; i<DimThreads; i++){
		SumP += PayOffs[i].GetPayOffs();
		SumP2 += PayOffs[i].GetPayOffs2();
	}
	double Price = SumP/N;

	return sqrt(abs(SumP2/N-Price*Price)/N);
};

__host__ void Statistics::Print(Statistics* PayOffs, int DimThreads){
	cout<<"Prezzo: "<<this->GetPrice(PayOffs, DimThreads)<<endl;
    cout<<"Errore MonteCarlo: "<<this->GetMCError(PayOffs, DimThreads)<<endl;
};

__host__ void Statistics::Print(std::string path, Statistics* PayOffs, int DimThreads){
	ofstream output;
	output.open(path.c_str());
	output<<"Prezzo: "<<this->GetPrice(PayOffs, DimThreads)<<endl;
    output<<"Errore MonteCarlo: "<<this->GetMCError(PayOffs, DimThreads)<<endl;
	cout<<"Risultati salvati nel file "<<path<<endl;
	output.close();
};

__host__ __device__ Statistics::Statistics(){
	_PayOffs=0;
	_PayOffs2=0;
	_Cont=0;
};

__host__ __device__ void Statistics::AddValue(double payoff){
	_PayOffs+=payoff;
	_PayOffs2+=payoff*payoff;
	_Cont=_Cont+1;
};

__host__ __device__ double Statistics::GetPayOffs(){
	return _PayOffs;
};

__host__ __device__ double Statistics::GetPayOffs2(){
	return _PayOffs2;
};

__host__ __device__ int Statistics::GetCont(){
	return _Cont;
};

__host__ __device__ void Statistics::Reset(){
	_PayOffs=0;
	_PayOffs2=0;
	_Cont=0;
};
