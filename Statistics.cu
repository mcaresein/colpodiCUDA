#include "Statistics.h"
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

__host__ HostStatistics::HostStatistics(DevStatistics* PayOffs, int DimThreads, int DimStreams){

	double SumP=0, SumP2=0;
	int N=DimThreads*DimStreams;

	for(int i=0; i<DimThreads; i++){
		SumP += PayOffs[i].GetPayOffs();
		SumP2 += PayOffs[i].GetPayOffs2();
	}
	_Price=SumP/N;
	_MCError=sqrt(abs(SumP2/N-_Price*_Price)/N);  //Ci va "abs"??
};

__host__ double HostStatistics::GetPrice(){
	return _Price;
};

__host__ double HostStatistics::GetMCError(){
	return _MCError;

};

__host__ void HostStatistics::Print(){
	cout<<"Prezzo: "<<this->GetPrice()<<endl;
    cout<<"Errore MonteCarlo: "<<this->GetMCError()<<endl;
};

__host__ void HostStatistics::Print(std::string path){
	ofstream output;
	output.open(path.c_str());
	output<<"Prezzo: "<<this->GetPrice()<<endl;
    output<<"Errore MonteCarlo: "<<this->GetMCError()<<endl;
	output.close();
};

__host__ __device__ DevStatistics::DevStatistics(){
	_PayOffs=0;
	_PayOffs2=0;
};

__host__ __device__ void DevStatistics::AddValue(double payoff){
	_PayOffs+=payoff;
	_PayOffs2+=payoff*payoff;
};

__host__ __device__ double DevStatistics::GetPayOffs(){
	return _PayOffs;
};

__host__ __device__ double DevStatistics::GetPayOffs2(){
	return _PayOffs2;
};
