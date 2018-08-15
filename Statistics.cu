#include "Statistics.h"
#include <cmath>

using namespace std;

__host__ __device__ Statistics::Statistics(){
	_Cumulant=0;
	_Cumulant2=0;
	_Cont=0;
};

__host__ __device__ void Statistics::AddValue(double value){
	_Cumulant+=value;
	_Cumulant2+=value*value;
	_Cont=_Cont+1;
};

__host__ __device__ double Statistics::GetCumulant(){
	return _Cumulant;
};

__host__ __device__ double Statistics::GetCumulant2(){
	return _Cumulant2;
};

__host__ __device__ int Statistics::GetCont(){
	return _Cont;
};

__host__ __device__ void Statistics::Reset(){
	_Cumulant=0;
	_Cumulant2=0;
	_Cont=0;
};

__host__ double Statistics::GetMean(){
	return this->GetCumulant()/this->GetCont();
};

__host__ double Statistics::GetStDev(){
	int N=this->GetCont();
	return sqrt(abs(this->GetCumulant2()/N-this->GetMean()*this->GetMean())/N);
};

__host__ Statistics Statistics::operator+(const Statistics& statistic){
	Statistics _statistic;
	_statistic._Cumulant=this->_Cumulant+statistic._Cumulant;
	_statistic._Cumulant2=this->_Cumulant2+statistic._Cumulant2;
	_statistic._Cont=this->_Cont+statistic._Cont;
	return _statistic;
};
