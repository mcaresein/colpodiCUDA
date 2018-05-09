#include <iostream>
#include <cmath>
#include "randomgen.h"

using namespace std;

__global__ void Kernel_old(unsigned int* S, double* W, int n){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	Randomgen obj(S[4*i],S[4*i+1],S[4*i+2],S[4*i+3]);

	for(int j=0; j<n; j++){
		W[i*n+j]=obj.Rand();
	}
}

__global__ void Kernel(unsigned int* S, double* W, int n){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	Randomgen obj(S[4*i],S[4*i+1],S[4*i+2],S[4*i+3]);

	for(int j=i; j<n*stride; j+=stride){
		W[j]=obj.Rand();
	}
}

__global__ void Kernel_test(unsigned int* S, double* W, double* test1, double* test2, int n){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	Randomgen obj(S[4*i],S[4*i+1],S[4*i+2],S[4*i+3]);

	for(int j=i; j<n*stride; j+=stride){
		W[j]=obj.Rand();
		__syncthreads();
		test1[j]=W[j]*W[(j+1)%stride];
	}
	for(int j=0; j<n; j++){
		test2[i+j*stride]=W[i+j*stride]*W[(i+j*stride+stride)%(stride*n)];
	}

}

int main(){
	srand(17);
	unsigned int *_S;
	double *_W;
	double *_t1;
	double *_t2;

	int streams=1000;
	int threads=4086;
	int dim_vec=streams*threads;

	unsigned int *S = new unsigned int[4*threads];
	double *W = new double[dim_vec];
	double *t1=new double[dim_vec];
	double *t2=new double[dim_vec];

	size_t sizeS = 4*threads * sizeof(unsigned int);
	size_t sizeW = dim_vec * sizeof(double);

	cudaMalloc((void**)& _S, sizeS);
	cudaMalloc((void**)& _W, sizeW);
	cudaMalloc((void**)& _t1, sizeW);
	cudaMalloc((void**)& _t2, sizeW);

	for(int i=0; i<4*threads; i++){
		S[i]=rand()+128;
	}

	cudaMemcpy(_S, S, sizeS, cudaMemcpyHostToDevice);

	int blockSize=512;
	int gridSize = (threads + blockSize - 1) / blockSize;

	Kernel_test<<<gridSize, blockSize>>>(_S, _W, _t1, _t2, streams);

	cudaMemcpy(W, _W, sizeW, cudaMemcpyDeviceToHost);
	cudaMemcpy(t1, _t1, sizeW, cudaMemcpyDeviceToHost);
	cudaMemcpy(t2, _t2, sizeW, cudaMemcpyDeviceToHost);

	double sum_t1=0;
	double var1=0;
	double sum_t2=0;
	double var2=0;

	for(int i=0; i<dim_vec; i++){
		var1=var1+(t1[i]-0.25)*(t1[i]-0.25);
		var2=var2+(t2[i]-0.25)*(t2[i]-0.25);
		sum_t1=sum_t1+t1[i];
		sum_t2=sum_t2+t2[i];
	}

	double sigma1=sqrt(var1/dim_vec);
	double sigma2=sqrt(var2/dim_vec);

	double avg1=sum_t1/dim_vec;
	double avg2=sum_t2/dim_vec;

	cout<<"Valore medio 1: "<<avg1<<endl;
	cout<<"Valore medio 2: "<<avg2<<endl;
	cout<<"Sigma 1: "<<sigma1<<endl;
	cout<<"Sigma 2: "<<sigma2<<endl;
	cout<<"Discrepanza 1: "<<abs(avg1-0.25)/sigma1<<endl;
	cout<<"Discrepanza 2: "<<abs(avg2-0.25)/sigma2<<endl;

	cudaFree(_S);
	cudaFree(_W);
	cudaFree(_t1);
	cudaFree(_t2);

	delete[] S;
	delete[] W;
	delete[] t1;
	delete[] t2;

    return 0;
}
