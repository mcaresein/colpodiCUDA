#include "Statistics.h"
#include <cmath>

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
