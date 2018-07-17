#include "Statistics.h"
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;



__host__ __device__ Statistics::Statistics(){
	_Cumulant=0;
	_Cumulant2=0;
	_Cont=0;
};

__host__ __device__ void Statistics::AddValue(double value){
	_Cumulant+=value;
	_Cumulant2+=value*value;
	_Cont=_Cont+1;
};

__host__ __device__ double Statistics::GetCumulant(){
	return _Cumulant;
};

__host__ __device__ double Statistics::GetCumulant2(){
	return _Cumulant2;
};

__host__ __device__ int Statistics::GetCont(){
	return _Cont;
};

__host__ __device__ void Statistics::Reset(){
	_Cumulant=0;
	_Cumulant2=0;
	_Cont=0;
};
__host__ double Statistics::GetMean(){
	return this->GetCumulant()/this->GetCont();
};
__host__ double Statistics::GetStDev(){
	int N=this->GetCont();
	return sqrt(abs(this->GetCumulant2()/N-this->GetMean()*this->GetMean())/N);
};

__host__ Statistics Statistics::operator+(const Statistics& statistic){
	Statistics _statistic;
	_statistic._Cumulant=this->_Cumulant+statistic._Cumulant;
	_statistic._Cumulant2=this->_Cumulant2+statistic._Cumulant2;
	_statistic._Cont=this->_Cont+statistic._Cont;
	return _statistic;
};
__host__ void Statistics::Print(double MaturityDate, double Drift){
	cout<<"Prezzo: "<<this->GetMean()*exp(-MaturityDate*Drift) <<endl;
  cout<<"Errore MonteCarlo: "<<this->GetStDev()<<endl;
}; 

__host__ void Statistics::Print(double MaturityDate, double Drift,std::string path){
	ofstream output;
	output.open(path.c_str());
	output<<"Prezzo: "<<this->GetMean()*exp(-MaturityDate*Drift)<<endl;
  output<<"Errore MonteCarlo: "<<this->GetStDev()<<endl;
	cout  <<"Risultati salvati nel file "<<path<<endl;
	output.close();
};
/*
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
*/
