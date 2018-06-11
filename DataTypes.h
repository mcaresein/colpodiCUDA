/*###############################
# Tipi di dati utilizzati nello #
# algoritmo.                    #
###############################*/

#ifndef _DataTypes_h_
#define _DataTypes_h_

struct Seed{
    unsigned int S1, S2, S3, S4;
};

struct MarketData{
    double Volatility, Drift, EquityInitialPrice;
};

struct OptionData{
    double TInitial, MaturityDate, StrikePrice, B, N, K;
    int NumberOfDatesToSimulate;
};

struct SimulationParameters{
    int OptionType;
    bool EulerApprox;
};

#endif
