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

class DevStatistics{
public:
	__host__ __device__ DevStatistics();
	__host__ __device__ void AddValue(double payoff);
	__host__ __device__ double GetPayOffs();
	__host__ __device__ double GetPayOffs2();
private:
	double _PayOffs;
	double _PayOffs2;
};

class HostStatistics {
public:
	__host__ HostStatistics(DevStatistics*, int, int);
	__host__ double GetPrice();
	__host__ double GetMCError();
	__host__ void Print();
	__host__ void Print(std::string);
private:
	double _Price;
	double _MCError;
};

#endif
