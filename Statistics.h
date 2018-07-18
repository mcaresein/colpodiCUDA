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
	__host__ __device__ void AddValue(double value);
	__host__ __device__ double GetCumulant();
  __host__ __device__ double GetCumulant2();
	__host__ __device__ int GetCont();
	__host__ __device__ void Reset();
	__host__ double GetMean();
	__host__ double GetStDev();
	__host__ Statistics operator+(const Statistics& statistic);
	__host__ void Print(double MaturityDate,double Drift);
	__host__ void Print(double MaturityDate, double Drift,std::string);

private:
	double _Cumulant;
	double _Cumulant2;
	int _Cont;  //contatore per la lunghezza del vettore per ogni thread. Cont=Streams/Threads
};

#endif
