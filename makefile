all: main

main: random_numbers.o randomgen.o
	nvcc -arch=sm_20 random_numbers.o randomgen.o -o prog

random_numbers.o: random_numbers.cu randomgen.h
	nvcc -arch=sm_20 -dc random_numbers.cu -o random_numbers.o -I.

randomgen.o: randomgen.cu randomgen.h
	nvcc -arch=sm_20 -dc randomgen.cu -o randomgen.o -I.
	
