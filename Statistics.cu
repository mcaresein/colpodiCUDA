#include "Statistics.h"
#include <cmath>

//## Metodo che implementa il calcolo del prezzo mediato su tutti gli scenari montecarlo e errore montecarlo associato. ##

__host__ Statistics::Statistics(float* Values, float* Values2, int DimThreads, int DimStreams){

	float SumP=0, SumP2=0;
	int N=DimThreads*DimStreams;

	for(int i=0; i<DimThreads; i++){
		SumP += Values[i];
		SumP2 += Values2[i];
	}
	_Price=SumP/N;
	_MCError=sqrt((SumP2/N-_Price*_Price)/N);
};

__host__ float Statistics::GetPrice(){
	return _Price;
};

__host__ float Statistics::GetMCError(){
	return _MCError;

};
