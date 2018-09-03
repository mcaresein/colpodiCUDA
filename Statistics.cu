#include "Statistics.h"
#include <cmath>

__host__ __device__ Statistics::Statistics(){
	_Sum=0;
	_Sum2=0;
	_Count=0;
};

__host__ __device__ void Statistics::AddValue(double value){
	_Sum+=value;
	_Sum2+=value*value;
	_Count=_Count+1;
};

__host__ __device__ void Statistics::Reset(){
	_Sum=0;
	_Sum2=0;
	_Count=0;
};

__host__ double Statistics::GetMean(){
	return _Sum/_Count;
};

__host__ double Statistics::GetStDev(){
	return sqrt(abs(_Sum2/_Count-this->GetMean()*this->GetMean())/_Count);
};

__host__ Statistics Statistics::operator+(const Statistics& statistic){
	Statistics _statistic;
	_statistic._Sum=this->_Sum+statistic._Sum;
	_statistic._Sum2=this->_Sum2+statistic._Sum2;
	_statistic._Count=this->_Count+statistic._Count;
	return _statistic;
};
