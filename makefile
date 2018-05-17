all: main

main: main.o RandomGenerator.o StocasticProcess.o UnderlyingPath.o PayOff.o Pricer.o Statistics.o
	nvcc -gencode arch=compute_20,code=sm_20 main.o RandomGenerator.o StocasticProcess.o UnderlyingPath.o PayOff.o Pricer.o Statistics.o -o prog

main.o: main.cu Pricer.h Statistics.h DataTypes.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc main.cu -o main.o -I.

RandomGenerator.o: RandomGenerator.cu RandomGenerator.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc RandomGenerator.cu -o RandomGenerator.o -I.
	
StocasticProcess.o: StocasticProcess.cu StocasticProcess.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc StocasticProcess.cu -o StocasticProcess.o -I.

UnderlyingPath.o: UnderlyingPath.cu UnderlyingPath.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc UnderlyingPath.cu -o UnderlyingPath.o -I.

PayOff.o: PayOff.cu PayOff.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc PayOff.cu -o PayOff.o -I.

Pricer.o: Pricer.cu Pricer.h RandomGenerator.h StocasticProcess.h UnderlyingPath.h PayOff.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc Pricer.cu -o Pricer.o -I.

Statistics.o: Statistics.cu Statistics.h
	nvcc -gencode arch=compute_20,code=sm_20 -dc Statistics.cu -o Statistics.o -I.

