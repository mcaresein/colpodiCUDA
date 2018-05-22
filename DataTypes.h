/*###############################
# Tipi di dati utilizzati nello #
# algoritmo.                    #
###############################*/

#ifndef _Seed_h_
#define _Seed_h_

struct Seed{
    unsigned int S1, S2, S3, S4;
};

struct MarketData{
    double Volatility, Drift, SInitial;
};

struct OptionData{
    double TInitial, TFinal, StrikePrice;
    int NSteps;
};

#endif
