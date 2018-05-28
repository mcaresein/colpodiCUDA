#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void Reader(MarketData &MarketInput, OptionData &OptionInput, int &THREADS, int &STREAMS){
    fstream file("DATA/input.conf");
    string line;
    double data[9];
    int i=0;
    while (getline(file, line)){
        if (line[0] == '#') continue;
        data[i]=atof(line.c_str());
        i++;
    }
    file.close();

    THREADS=data[0];
    STREAMS=data[1];

    MarketInput.Volatility=data[2];
    MarketInput.Drift=data[3];
    MarketInput.EquityInitialPrice=data[4];

    OptionInput.TInitial=data[5];
    OptionInput.MaturityDate=data[6];
    OptionInput.NumberOfDatesToSimulate=data[7];
    OptionInput.StrikePrice=data[8];
};

void GetSeeds(Seed* SeedVector, int THREADS){

    srand(17*17);

    for(int i=0; i<THREADS; i++){
        SeedVector[i].S1=rand()%(UINT_MAX-128)+128;
        SeedVector[i].S2=rand()%(UINT_MAX-128)+128;
        SeedVector[i].S3=rand()%(UINT_MAX-128)+128;
        SeedVector[i].S4=rand();
    }
};
