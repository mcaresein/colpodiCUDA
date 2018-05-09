class Randomgen{
public:
    __host__ __device__ Randomgen(int Sa, int Sb, int Sc, int Sd);
    __host__ __device__ unsigned int LCGStep(unsigned int seed, unsigned int a, unsigned long b);
    __host__ __device__ unsigned int TausStep(unsigned int seed, unsigned int K1, unsigned int K2, unsigned int K3, unsigned long M);
    __host__ __device__ double Rand();
    __host__ __device__ double Gauss();
private:
    unsigned int _Sa, _Sb, _Sc, _Sd;
};
