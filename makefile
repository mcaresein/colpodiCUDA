all: pricer pricer_comp

pricer: main.o RandomGenerator.o RandomGeneratorCombined.o StochasticProcess.o MonteCarloPath.o Option.o MonteCarloPricer.o Statistics.o
	nvcc -gencode arch=compute_20,code=sm_20 main.o RandomGenerator.o RandomGeneratorCombined.o StochasticProcess.o MonteCarloPath.o Option.o MonteCarloPricer.o Statistics.o -o pricer

main.o: main.cu KernelFunctions.cu Utilities.cu MonteCarloPricer.h Statistics.h Seed.h OptionData.h SimulationParameters.h GPUData.h MarketData.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc main.cu -o main.o -I.

pricer_comp: main_comp.o RandomGenerator.o RandomGeneratorCombined.o StochasticProcess.o MonteCarloPath.o Option.o MonteCarloPricer.o Statistics.o 
	nvcc -gencode arch=compute_20,code=sm_20 main_comp.o RandomGenerator.o RandomGeneratorCombined.o StochasticProcess.o MonteCarloPath.o Option.o MonteCarloPricer.o Statistics.o -o pricer_comp

main_comp.o: main_comp.cu MonteCarloPricer.h Statistics.h Seed.h OptionData.h SimulationParameters.h GPUData.h MarketData.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc main_comp.cu -o main_comp.o -I.

RandomGenerator.o: RandomGenerator.cu  RandomGenerator.h RandomGeneratorCombined.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc RandomGenerator.cu -o RandomGenerator.o -I.

RandomGeneratorCombined.o: RandomGeneratorCombined.cu  RandomGenerator.h RandomGeneratorCombined.h
	 nvcc -gencode arch=compute_20,code=sm_20 -dc RandomGeneratorCombined.cu -o RandomGeneratorCombined.o -I.

StochasticProcess.o: StochasticProcess.cu StochasticProcess.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc StochasticProcess.cu -o StochasticProcess.o -I.

MonteCarloPath.o: MonteCarloPath.cu MonteCarloPath.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc MonteCarloPath.cu -o MonteCarloPath.o -I.

Option.o: Option.cu Option.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc Option.cu -o Option.o -I.

MonteCarloPricer.o: MonteCarloPricer.cu MonteCarloPricer.h RandomGenerator.h StochasticProcess.h MonteCarloPath.h Option.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc MonteCarloPricer.cu -o MonteCarloPricer.o -I.

Statistics.o: Statistics.cu Statistics.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc Statistics.cu -o Statistics.o -I. 

clean:
	rm *.o

