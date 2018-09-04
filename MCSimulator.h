#include <iostream>
#include <cstdio>
#include <ctime>
#include <fstream>
#include "Statistics.h"
#include "Seed.h"
#include "MarketData.h"
#include "OptionData.h"
#include "GPUData.h"
#include "SimulationParameters.h"
#include "KernelFunctions.h"
#include "RandomGenerator.h"
#include "RandomGeneratorCombined.h"


using namespace std;

class MCSimulator{
private:
    string _InputFile, _OutputFile;
    int _Seed;
    bool CPUComparison;

    MarketData MarketInput;
    OptionDataContainer OptionInput;
    SimulationParameters Parameters;
    GPUData GPUInput;

    Statistics* PayOffsGPU;
    Statistics* PayOffsCPU;
    Seed *SeedVector;
    Statistics* _PayOffsGPU;
    Seed *_SeedVector;

    size_t sizeSeedVector;
    size_t sizeDevStVector;

    int gridSize;
    cudaEvent_t start, stop;
    float GPUTime;
    double CPUTime;
    clock_t startcpu;

    Statistics FinalStatisticsGPU;
    Statistics FinalStatisticsCPU;

    double GPUPrice;
    double CPUPrice;
    double GPUError;
    double CPUError;

    void GetSeeds(Seed* SeedVector, int THREADS, int seed);
    void MemoryAllocationCPU(Statistics** PayOffsCPU, int THREADS);
    void MemoryAllocationGPU(Statistics** PayOffsGPU, Seed** SeedVector, Statistics** _PayOffsGPU, Seed** _SeedVector, size_t sizeSeedVector, size_t sizeDevStVector, int THREADS);
    void MemoryDeallocationGPU(Statistics* PayOffsGPU, Seed* SeedVector, Statistics* _PayOffsGPU, Seed* _SeedVector);
    void MemoryDeallocationCPU(Statistics* PayOffsCPU);
    void PrintActualizedPrice(Statistics Stat ,double MaturityDate, double Drift, ofstream& output);
    void PrintComparison(Statistics FinalStatisticsGPU, Statistics FinalStatisticsCPU, ofstream& output);
    void Reader(string InputFile, MarketData &MarketInput, OptionDataContainer &OptionInput, GPUData &GPUInput, SimulationParameters &Parameters, bool &CPUComparison, ofstream& output);

public:
    MCSimulator(int Seed, string InputFile, string OutputFile);
    int main();
    int RegressionTest();
};
