/*#############################################
# Classe che implementa i metodi per ottenere #
# prezzo dell'opzione mediato su tutti gli    #
# scenari montecarlo (GetMean) e l'errore    #
# montecarlo associato (GetStDev).          #
#############################################*/

#ifndef _Statistics_h_
#define _Statistics_h_

class Statistics{
public:
	__host__ __device__ Statistics();
	__host__ __device__ void AddValue(double value);
	__host__ __device__ void Reset();
	__host__ double GetMean();
	__host__ double GetStDev();
	__host__ Statistics operator+(const Statistics& statistic);

private:
	double _Sum;
	double _Sum2;
	int _Count;
};

#endif
