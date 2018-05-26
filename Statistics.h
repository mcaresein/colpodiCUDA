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

class Statistics {
public:
	__host__ Statistics(double*, double*, int, int);
	__host__ double GetPrice();
	__host__ double GetMCError();
	__host__ void Print();
	__host__ void Print(std::string);
private:

	double _Price;
	double _MCError;
};

#endif
