all: main

main: main.o RandomGenerator.o StocasticProcess.o UnderlyingPath.o Option.o MonteCarloPricer.o Statistics.o
	nvcc -gencode arch=compute_20,code=sm_20 main.o RandomGenerator.o StocasticProcess.o UnderlyingPath.o Option.o MonteCarloPricer.o Statistics.o -o Pricer

main.o: main.cu MonteCarloPricer.h Statistics.h DataTypes.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc main.cu -o main.o -I.

RandomGenerator.o: RandomGenerator.cu RandomGenerator.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc RandomGenerator.cu -o RandomGenerator.o -I.

StocasticProcess.o: StocasticProcess.cu StocasticProcess.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc StocasticProcess.cu -o StocasticProcess.o -I.

UnderlyingPath.o: UnderlyingPath.cu UnderlyingPath.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc UnderlyingPath.cu -o UnderlyingPath.o -I.

Option.o: Option.cu Option.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc Option.cu -o Option.o -I.

MonteCarloPricer.o: MonteCarloPricer.cu MonteCarloPricer.h RandomGenerator.h StocasticProcess.h UnderlyingPath.h Option.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc MonteCarloPricer.cu -o MonteCarloPricer.o -I.

Statistics.o: Statistics.cu Statistics.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc Statistics.cu -o Statistics.o -I.
