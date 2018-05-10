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

__global__ void Kernel_test(unsigned int* S, double* W, double* test, double* sum, double* sum2, int n){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	Randomgen obj(S[4*i],S[4*i+1],S[4*i+2],S[4*i+3]);
	int cont=1;
	double old_value=0;
	double sumt=0, sum2t=0;

	for(int j=i; j<n*stride; j+=stride){
		W[j]=obj.Rand();
		__syncthreads();
		test[j]=W[j]*W[(j+1)%(stride*cont)];	//controllo inter-thread
		sumt+=W[j]*old_value;					//controllo intra-thread
		sum2t+=W[j]*old_value*W[j]*old_value;
		old_value=W[j];
		cont++;
	}
	sum[i]=sumt;
	sum2[i]=sum2t;
}

int main(){
	srand(17);

	unsigned int *_S;
	double *_W;
	double *_test;
	double *_sum;
	double *_sum2;

	int streams=1000;
	int threads=1024;
	int dim_vec=streams*threads;

	unsigned int *S = new unsigned int[4*threads];
	double *W = new double[dim_vec];
	double *test=new double[dim_vec];
	double *sum=new double[threads];
	double *sum2=new double[threads];

	size_t sizeS = 4*threads * sizeof(unsigned int);
	size_t sizeW = dim_vec * sizeof(double);
	size_t sizeSum = threads * sizeof(double);

	cudaMalloc((void**)& _S, sizeS);
	cudaMalloc((void**)& _W, sizeW);
	cudaMalloc((void**)& _test, sizeW);
	cudaMalloc((void**)& _sum, sizeSum);
	cudaMalloc((void**)& _sum2, sizeSum);

	for(int i=0; i<4*threads; i++){
		S[i]=rand()+128;
	}

	cudaMemcpy(_S, S, sizeS, cudaMemcpyHostToDevice);

	int blockSize=512;
	int gridSize = (threads + blockSize - 1) / blockSize;

	Kernel_test<<<gridSize, blockSize>>>(_S, _W, _test, _sum, _sum2, streams);

	cudaMemcpy(W, _W, sizeW, cudaMemcpyDeviceToHost);
	cudaMemcpy(test, _test, sizeW, cudaMemcpyDeviceToHost);
	cudaMemcpy(sum, _sum, sizeSum, cudaMemcpyDeviceToHost);
	cudaMemcpy(sum2, _sum2, sizeSum, cudaMemcpyDeviceToHost);

	double avg1=0, avg2=0, avg1_temp=0, avg2_temp;
	double var1=0;

	for(int i=0; i<threads; i++){
		avg1_temp+=sum[i]/streams;
		avg2_temp+=sum2[i]/streams;
	}

	avg1=avg1_temp/threads;
	avg2=avg2_temp/threads;

	var1= avg2 - avg1*avg1;

	double discrepanza=(avg1-0.25)/sqrt(var1/threads);

	cout<<"avg: "<<avg1<<endl;
	cout<<"var: "<<var1<<endl;
	cout<<"dis: "<<discrepanza<<endl;

	double* sum_interT = new double[streams];
	double* sum_interT2 = new double[streams];

	for(int i=0; i<streams; i++){
		sum_interT[i]=0;
		sum_interT2[i]=0;
	}

	for(int i=0; i<streams; i++){
		for(int j=0; j<threads; j++){
			sum_interT[i]+=test[j+threads*i];
			sum_interT2[i]+=test[j+threads*i]*test[j+threads*i];
		}
		sum_interT[i]= sum_interT[i]/threads;
		sum_interT2[i]= sum_interT2[i]/threads;
	}

    double avg_interT_temp=0, avg_interT=0;
	double avg_interT2_temp=0, avg_interT2=0;


	for(int i=0; i<streams; i++){
		avg_interT_temp+=sum_interT[i];
		avg_interT2_temp+=sum_interT2[i];
	}

	avg_interT = avg_interT_temp/streams;
	avg_interT2 = avg_interT2_temp/streams;

	double var_inter2= avg_interT2 - avg_interT*avg_interT;
	double discrepanza_interT= (avg_interT - 0.25)/sqrt(var_inter2/streams);

	cout<<"avg: "<<avg_interT<<" var: "<<var_inter2<<" dis: "<<discrepanza_interT<<endl;


/*	double sum_t1=0;
	double var1=0;
	double sum_t2=0;
	double var2=0;

	for(int i=0; i<dim_vec; i++){
		var1=var1+(t1[i]-0.25)*(t1[i]-0.25);
	//	var2=var2+(t2[i]-0.25)*(t2[i]-0.25);
		sum_t1=sum_t1+t1[i];
	//	sum_t2=sum_t2+t2[i];
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
*/
	cudaFree(_S);
	cudaFree(_W);
	cudaFree(_test);
	cudaFree(_sum);
	cudaFree(_sum2);

	delete[] S;
	delete[] W;
	delete[] test;
	delete[] sum;
	delete[] sum2;

    return 0;
}
