#ifndef _KernelFunctions_cu_
#define _KernelFunctions_cu_

#define RE_EXTRACTION_BOX_MULLER false

#include "MonteCarloPricer.h"
#include "Statistics.h"
#include "RandomGenerator.h"
#include "RandomGeneratorCombined.h"
#include "RandomGeneratorCombinedGaussian.h"
#include "RandomGeneratorCombinedBimodal.h"
#include "Gaussian.h"
#include "Seed.h"
#include "MarketData.h"
#include "OptionData.h"
#include "SimulationParameters.h"
#include "Option.h"
#include "UnderlyingAnagraphy.h"
#include "UnderlyingPrice.h"

/*############################ Kernel Functions ############################*/

__host__ __device__ void TrueKernel(Seed* SeedVector, Statistics* PayOffs, int streams, MarketData MarketInput, OptionDataContainer OptionInput, SimulationParameters Parameters, int cont);

__global__ void Kernel(Seed* SeedVector, Statistics* PayOffs, int streams, MarketData MarketInput, OptionDataContainer OptionInput, SimulationParameters Parameters);

//## Funzione che gira su CPU che restituisce due vettori con sommme dei PayOff e dei PayOff quadrati. ##

__host__ void KernelSimulator(Seed* SeedVector, Statistics* PayOffs, int streams, MarketData MarketInput, OptionDataContainer OptionInput, SimulationParameters Parameters, int threads);

#endif
