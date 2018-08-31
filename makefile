pricer: main.o MCSimulator.o KernelFunctions.o RandomGenerator.o Gaussian.o RandomGeneratorCombined.o RandomGeneratorCombinedGaussian.o StochasticProcess.o MonteCarloPath.o Option.o MonteCarloPricer.o Statistics.o
	nvcc -gencode arch=compute_20,code=sm_20 main.o MCSimulator.o KernelFunctions.o RandomGenerator.o RandomGeneratorCombined.o RandomGeneratorCombinedGaussian.o Gaussian.o StochasticProcess.o MonteCarloPath.o Option.o MonteCarloPricer.o Statistics.o -o pricer

test: RegressionTest.o MCSimulator.o KernelFunctions.o RandomGenerator.o RandomGeneratorCombined.o RandomGeneratorCombinedGaussian.o Gaussian.o  StochasticProcess.o MonteCarloPath.o Option.o MonteCarloPricer.o Statistics.o
	nvcc -gencode arch=compute_20,code=sm_20 RegressionTest.o MCSimulator.o KernelFunctions.o RandomGenerator.o RandomGeneratorCombined.o RandomGeneratorCombinedGaussian.o Gaussian.o  StochasticProcess.o MonteCarloPath.o Option.o MonteCarloPricer.o Statistics.o -o test

RegressionTest.o: RegressionTest.cu
	nvcc -gencode arch=compute_20,code=sm_20 -dc RegressionTest.cu -o RegressionTest.o -I.

main.o: main.cu
	nvcc -gencode arch=compute_20,code=sm_20 -dc main.cu -o main.o -I.

MCSimulator.o: MCSimulator.cu MCSimulator.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc MCSimulator.cu -o MCSimulator.o -I.

Gaussian.o: Gaussian.cu Gaussian.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc Gaussian.cu -o Gaussian.o -I.

KernelFunctions.o: KernelFunctions.cu KernelFunctions.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc KernelFunctions.cu -o KernelFunctions.o -I.

RandomGenerator.o: RandomGenerator.cu  RandomGenerator.h 
	nvcc -gencode arch=compute_20,code=sm_20 -dc RandomGenerator.cu -o RandomGenerator.o -I.

RandomGeneratorCombined.o: RandomGeneratorCombined.cu  RandomGenerator.h RandomGeneratorCombined.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc RandomGeneratorCombined.cu -o RandomGeneratorCombined.o -I.

RandomGeneratorCombinedGaussian.o:RandomGeneratorCombinedGaussian.cu RandomGenerator.h RandomGeneratorCombined.h RandomGeneratorCombinedGaussian.h Gaussian.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc RandomGeneratorCombinedGaussian.cu -o RandomGeneratorCombinedGaussian.o -I.

StochasticProcess.o: StochasticProcess.cu StochasticProcess.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc StochasticProcess.cu -o StochasticProcess.o -I.

MonteCarloPath.o: MonteCarloPath.cu MonteCarloPath.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc MonteCarloPath.cu -o MonteCarloPath.o -I.

Option.o: Option.cu Option.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc Option.cu -o Option.o -I.

MonteCarloPricer.o: MonteCarloPricer.cu MonteCarloPricer.h RandomGenerator.h RandomGeneratorCombinedGaussian.h StochasticProcess.h MonteCarloPath.h Option.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc MonteCarloPricer.cu -o MonteCarloPricer.o -I.

Statistics.o: Statistics.cu Statistics.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc Statistics.cu -o Statistics.o -I.

clean:
	rm *.o
