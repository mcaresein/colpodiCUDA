#ifndef _OptionData_h_
#define _OptionData_h_

struct OptionDataContainer{
    double MaturityDate;
    int NumberOfFixingDate;
    double StrikePrice;
    int OptionType;
    double B, N, K;
};

struct OptionData{
    double MaturityDate;
    int NumberOfFixingDate;
    int OptionType;
    double* AdditionalParameters;
};

#endif
