#ifndef _Gaussian_h_
#define _Gaussian_h_

class Gaussian{
public:
    __host__ __device__ Gaussian(bool BoxMullerWithReExtraction);
    __host__ __device__ virtual double GetGaussianRandomNumber(double RN1, double RN2);

protected:
    double _SavedRandomNumber;
    bool _Status, _BoxMullerWithReExtraction;
};

#endif
