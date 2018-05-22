/*#############################################
# Classe che implementa i metodi per ottenere #
# prezzo dell'opzione mediato su tutti gli    #
# scenari montecarlo (GetPrice) e l'errore    #
# montecarlo associato (GetMCError).          #
#############################################*/

#ifndef _Statistics_h_
#define _Statistics_h_

class Statistics {
public:
	__host__ Statistics(double*, double*, int, int);
	__host__ double GetPrice();
	__host__ double GetMCError();
private:
	double _Price;
	double _MCError;
};

#endif
