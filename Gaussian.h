#ifndef _Gaussian_h_
#define _Gaussian_h_

#include "RandomGenerator.h"

class Gaussian{
public:
    __host__ __device__ Gaussian(bool BoxMullerWithReExtraction);
    __host__ __device__ double GetGaussianRandomNumber(RandomGenerator* Generator);

protected:
    double _SavedRandomNumber;
    bool _Status, _BoxMullerWithReExtraction;
};

#endif
