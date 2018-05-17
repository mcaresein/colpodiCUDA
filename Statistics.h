#ifndef _Statistics_h_
#define _Statistics_h_

class Statistics {
public:
	__host__ Statistics(float*, float*, int, int);
	__host__ float GetPrice();
	__host__ float GetMCError();
private:
	float _Price;
	float _MCError;
};

#endif
