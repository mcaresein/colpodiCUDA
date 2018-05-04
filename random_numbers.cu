#include <iostream>
#include "randomgen.h"

using namespace std;

__global__ void Kernel(unsigned int* S, double* W, int n){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	Randomgen obj(S[4*i],S[4*i+1],S[4*i+2],S[4*i+3]);

	for(int j=0; j<n; j++){
		W[i*n+j]=obj.Rand();
		__syncthreads();
	}
}

int main(){
	srand(17);
	unsigned int *_S;
	double *_W;

	int streams=100;
	int threads=1024;
	int dim_vec=streams*threads;

	unsigned int *S = new unsigned int[4*threads];
	double *W = new double[dim_vec];

	size_t sizeS = 4*threads * sizeof(unsigned int);
	size_t sizeW = dim_vec * sizeof(double);

	cudaMalloc((void**)& _S, sizeS);
	cudaMalloc((void**)& _W, sizeW);

	for(int i=0; i<4*threads; i++){
		S[i]=rand()+128;
	}

	cudaMemcpy(_S, S, sizeS, cudaMemcpyHostToDevice);

	int blockSize=512;
	int gridSize = (threads + blockSize - 1) / blockSize;

	Kernel<<<gridSize, blockSize>>>(_S, _W, streams);

	cudaMemcpy(W, _W, sizeW, cudaMemcpyDeviceToHost);

	double sum_over_streams;
	double sum_over_threads;

	for(int i=0; i<threads ;i++)
		for(int j=0; j<streams; j++)
			sum_over_streams=sum_over_streams+W[i*streams+j];


	cudaFree(_S);
	cudaFree(_W);

	delete[] S;
	delete[] W;

    return 0;
}
