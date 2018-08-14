#ifndef _UnderlyingPrice_h_
#define _UnderlyingPrice_h_

#include "UnderlyingAnagraphy.h"

struct UnderlyingPrice{
    double Price;
    UnderlyingAnagraphy* Anagraphy;
    __host__ __device__ UnderlyingPrice(UnderlyingAnagraphy* AnagraphyInput){
        Anagraphy=AnagraphyInput;
    };
};

#endif
