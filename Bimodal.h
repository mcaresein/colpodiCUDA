#ifndef _Bimodal_h_
#define _Bimodal_h_

#include "RandomGenerator.h"

class Bimodal {
public:
    __host__ __device__  double GetBimodalRandomNumber(RandomGenerator* Generator);
};

#endif
