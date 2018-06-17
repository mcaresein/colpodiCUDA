/*#############################################
# Classe che implementa i metodi per ottenere #
# prezzo dell'opzione mediato su tutti gli    #
# scenari montecarlo (GetPrice) e l'errore    #
# montecarlo associato (GetMCError).          #
#############################################*/

#ifndef _Statistics_h_
#define _Statistics_h_
#include <iostream>
#include <string>

class Statistics{
public:
	__host__ __device__ Statistics();
	__host__ __device__ void AddValue(double payoff);
	__host__ __device__ double GetPayOffs();
  __host__ __device__ double GetPayOffs2();
	__host__ __device__ int GetCont();
	__host__ __device__ void Reset();
	__host__ double GetPrice(Statistics* PayOffs, int DimThreads);
	__host__ double GetMCError(Statistics* PayOffs, int DimThreads);
	__host__ void Print(Statistics* PayOffs, int DimThreads);
	__host__ void Print(std::string, Statistics* PayOffs, int DimThreads);

private:
	double _PayOffs;
	double _PayOffs2;
	int _Cont;
};

#endif
