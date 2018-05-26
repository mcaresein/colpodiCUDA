#include "Statistics.h"
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

__host__ Statistics::Statistics(double* Values, double* Values2, int DimThreads, int DimStreams){

	double SumP=0, SumP2=0;
	int N=DimThreads*DimStreams;

	for(int i=0; i<DimThreads; i++){
		SumP += Values[i];
		SumP2 += Values2[i];
	}
	_Price=SumP/N;
	_MCError=sqrt((SumP2/N-_Price*_Price)/N);
};

__host__ double Statistics::GetPrice(){
	return _Price;
};

__host__ double Statistics::GetMCError(){
	return _MCError;

};

__host__ void Statistics::Print(){
	cout<<"Prezzo: "<<this->GetPrice()<<endl;
    cout<<"Errore MonteCarlo: "<<this->GetMCError()<<endl;
}

__host__ void Statistics::Print(std::string path){
	ofstream output;
	output.open(path.c_str());
	output<<"Prezzo: "<<this->GetPrice()<<endl;
    output<<"Errore MonteCarlo: "<<this->GetMCError()<<endl;
	output.close();
}
