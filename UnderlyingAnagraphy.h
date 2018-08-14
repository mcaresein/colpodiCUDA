#ifndef _UnderlyingAnagraphy_h_
#define _UnderlyingAnagraphy_h_

#include "MarketData.h"

struct UnderlyingAnagraphy{
    double Volatility;
    double Drift;
    __host__ __device__ UnderlyingAnagraphy(MarketData MarketInput){
        Volatility=MarketInput.Volatility;
        Drift=MarketInput.Drift;
    };
};

#endif
