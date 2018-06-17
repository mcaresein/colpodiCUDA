
#ifndef _OptionData_h_
#define _OptionData_h_

struct OptionData{
    double  MaturityDate;
    int NumberOfDatesToSimulate, EulerSubStep;
    double StrikePrice;
    int OptionTypeCallOrPut;
    double B, N, K;
};

/*
struct OptionData{
protected:
    double  MaturityDate;
    int NumberOfDatesToSimulate, EulerSubStep;
public:
    __host__ __device__ virtual double GetStrikePrice(){return 0;};
    __host__ __device__ virtual int GetOptionTypeCallOrPut(){return 0;};
    __host__ __device__ virtual double GetB(){return 0;};
    __host__ __device__ virtual double GetN(){return 0;};
    __host__ __device__ virtual double GetK(){return 0;};
    __host__ __device__ virtual double GetMaturityDate(){return MaturityDate;};
    __host__ __device__ virtual int GetNumberOfDatesToSimulate(){return NumberOfDatesToSimulate;};
    __host__ __device__ virtual int GetEulerSubStep(){return EulerSubStep;};
};

struct OptionDataForward: OptionData{
    OptionDataForward(double init_MaturityDate, int init_NumberOfDatesToSimulate, int init_EulerSubStep){
        MaturityDate=init_MaturityDate;
        NumberOfDatesToSimulate=init_NumberOfDatesToSimulate;
        EulerSubStep=init_EulerSubStep;
    }
};

struct OptionDataPlainVanilla: OptionData{
    double StrikePrice;
    int OptionTypeCallOrPut;
    OptionDataPlainVanilla(double init_MaturityDate, int init_NumberOfDatesToSimulate, int init_EulerSubStep, double init_StrikePrice, double init_OptionTypeCallOrPut){
        MaturityDate=init_MaturityDate;
        NumberOfDatesToSimulate=init_NumberOfDatesToSimulate;
        EulerSubStep=init_EulerSubStep;
        StrikePrice=init_StrikePrice;
        OptionTypeCallOrPut=init_OptionTypeCallOrPut;
    }
    __host__ __device__ double GetStrikePrice(){
        return StrikePrice;
    }
    __host__ __device__ int GetOptionTypeCallOrPut(){
        return OptionTypeCallOrPut;
    }
};

struct OptionDataAbsolutePerformanceBarrier: OptionData{
    double B, N, K;
    OptionDataAbsolutePerformanceBarrier(double init_MaturityDate, int init_NumberOfDatesToSimulate, int init_EulerSubStep, double init_B, double init_N, double init_K){
        MaturityDate=init_MaturityDate;
        NumberOfDatesToSimulate=init_NumberOfDatesToSimulate;
        EulerSubStep=init_EulerSubStep;
        B=init_B;
        N=init_N;
        K=init_K;
    }
    __host__ __device__ double GetB(){
        return B;
    }
    __host__ __device__ double GetK(){
        return K;
    }
    __host__ __device__ double GetN(){
        return N;
    }
};
*/
#endif
