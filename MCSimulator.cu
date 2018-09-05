#include "MCSimulator.h"

using namespace std;

//## Funzione main #############################################################

int MCSimulator::main(){
    ofstream output;
    output.open(_OutputFile.c_str());

    //## Inizializzazione parametri di mercato e opzione. ##########################

    Reader(_InputFile, MarketInput, OptionInput, GPUInput, Parameters, CPUComparison, output);

    //## Allocazione di memoria. ###################################################

    sizeSeedVector = GPUInput.Threads * sizeof(Seed);
    sizeDevStVector = GPUInput.Threads * sizeof(Statistics);

    MemoryAllocationGPU(& PayOffsGPU,  & SeedVector, & _PayOffsGPU, & _SeedVector, sizeSeedVector, sizeDevStVector, GPUInput.Threads);

    if(CPUComparison==true)
        MemoryAllocationCPU(& PayOffsCPU, GPUInput.Threads);

    //## Costruzione vettore dei seed. #############################################

    GetSeeds(SeedVector, GPUInput.Threads, _Seed);

    cudaMemcpy(_SeedVector, SeedVector, sizeSeedVector, cudaMemcpyHostToDevice);

    //## Calcolo dei PayOff su GPU. ################################################

    cout<<"GPU simulation..."<<endl;

    gridSize = (GPUInput.Threads + GPUInput.BlockSize - 1) / GPUInput.BlockSize;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    Kernel<<<gridSize, GPUInput.BlockSize>>>(_SeedVector, _PayOffsGPU, GPUInput.Streams, MarketInput, OptionInput, Parameters);
    cudaEventRecord(stop);

    cudaMemcpy(PayOffsGPU, _PayOffsGPU, sizeDevStVector, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&GPUTime, start, stop);

    //## Calcolo dei PayOff su CPU. ################################################

    if(CPUComparison==true){
        cout<<"CPU simulation..."<<endl;

        startcpu = clock();
        KernelSimulator(SeedVector, PayOffsCPU, GPUInput.Streams, MarketInput, OptionInput, Parameters, GPUInput.Threads);
        CPUTime = (clock() - startcpu ) / (double) CLOCKS_PER_SEC;
    }

    //## Statistica... #####################################################

    for (int i=0; i<GPUInput.Threads; i++)
        FinalStatisticsGPU=FinalStatisticsGPU+PayOffsGPU[i];


    if(CPUComparison==true){
        for (int i=0; i<GPUInput.Threads; i++)
            FinalStatisticsCPU=FinalStatisticsCPU+PayOffsCPU[i];
    }

    //## Calcolo e stampa su file dei valori. ######################################

    cout<<endl<<"GPU values:"<<endl;
    output<<endl<<"GPU values:"<<endl;
    PrintActualizedPrice(FinalStatisticsGPU, OptionInput.MaturityDate, MarketInput.Drift, output);
    cout<<"Simulation time: "<<GPUTime<<" ms"<<endl<<endl;
    output<<"Simulation time: "<<GPUTime<<" ms"<<endl<<endl;

    if(CPUComparison==true){
        cout<<"CPU values:"<<endl;
        output<<"CPU values:"<<endl;
        PrintActualizedPrice(FinalStatisticsCPU, OptionInput.MaturityDate, MarketInput.Drift, output);
        cout<<"Simulation time: "<<CPUTime*1000<<" ms"<<endl<<endl;
        output<<"Simulation time: "<<CPUTime*1000<<" ms"<<endl<<endl;
    }

    //## Controllo #################################################################
    if(CPUComparison==true)
        PrintComparison(FinalStatisticsGPU, FinalStatisticsCPU, output);

    //## Liberazione memoria. ######################################################

    MemoryDeallocationGPU(PayOffsGPU, SeedVector, _PayOffsGPU, _SeedVector);
    if(CPUComparison==true)
        MemoryDeallocationCPU(PayOffsCPU);

    output.close();

    return 0;
}

//##############################################################################

//## Funzioni e utilities ######################################################

MCSimulator::MCSimulator(int Seed, string InputFile, string OutputFile){
    _InputFile=InputFile;
    _OutputFile=OutputFile;
    _Seed=Seed;
    GPUTime = 0;
}

void MCSimulator::GetSeeds(Seed* SeedVector, int THREADS, int seed){

    srand(seed);

    for(int i=0; i<THREADS; i++){
        do{
            SeedVector[i].S1=rand();
        }while(SeedVector[i].S1<128);
        do{
            SeedVector[i].S2=rand();
        }while(SeedVector[i].S2<128);
        do{
            SeedVector[i].S3=rand();
        }while(SeedVector[i].S3<128);
        SeedVector[i].S4=rand();
    }
};

void MCSimulator::MemoryAllocationCPU(Statistics** PayOffsCPU, int THREADS){
    *PayOffsCPU = new Statistics[THREADS];
};

void MCSimulator::MemoryAllocationGPU(Statistics** PayOffsGPU, Seed** SeedVector, Statistics** _PayOffsGPU, Seed** _SeedVector, size_t sizeSeedVector, size_t sizeDevStVector, int THREADS){
    cout<<"Memory allocation..."<<endl;
    *PayOffsGPU=new Statistics[THREADS];
    *SeedVector= new Seed[THREADS];

    cudaMalloc((void**)& *_PayOffsGPU, sizeDevStVector);
    cudaMalloc((void**)& *_SeedVector, sizeSeedVector);
};

void MCSimulator::MemoryDeallocationCPU(Statistics* PayOffsCPU){
    delete[] PayOffsCPU;
};

void MCSimulator::MemoryDeallocationGPU(Statistics* PayOffsGPU, Seed* SeedVector, Statistics* _PayOffsGPU, Seed* _SeedVector){
    delete[] PayOffsGPU;
    delete[] SeedVector;

    cudaFree(_PayOffsGPU);
    cudaFree(_SeedVector);

    cudaDeviceReset();
};

void MCSimulator::PrintActualizedPrice(Statistics Stat ,double MaturityDate, double Drift, ofstream& output){
    cout<<"Price: "<<Stat.GetMean()*exp(-MaturityDate*Drift) <<endl;
    cout<<"MC error: "<<Stat.GetStDev()<<endl;

    output<<"Price: "<<Stat.GetMean()*exp(-MaturityDate*Drift)<<endl;
    output<<"MC error: "<<Stat.GetStDev()<<endl;
};

void MCSimulator::PrintComparison(Statistics FinalStatisticsGPU, Statistics FinalStatisticsCPU, ofstream& output){
    GPUPrice=FinalStatisticsGPU.GetMean();
    CPUPrice=FinalStatisticsCPU.GetMean();
    GPUError=FinalStatisticsGPU.GetStDev();
    CPUError=FinalStatisticsCPU.GetStDev();

    if(GPUPrice==CPUPrice){
        cout<<"Prices are equivalent!"<<endl;
        output<<"Prices are equivalent!"<<endl;
    }

    else{
        cout<<"Prices are NOT equivalent!"<<endl;
        cout<<"Discrepancy: "<<GPUPrice-CPUPrice<<endl;
        output<<"Prices are NOT equivalent!"<<endl;
        output<<"Discrepancy: "<<GPUPrice-CPUPrice<<endl;
    }
    if(GPUError==CPUError){
        cout<<"Errors are equivalent!"<<endl;
        output<<"Errors are equivalent!"<<endl;
    }
    else{
        cout<<"Errors are NOT equivalent!"<<endl;
        cout<<"Discrepancy: "<<GPUError-CPUError<<endl;
        output<<"Errors are NOT equivalent!"<<endl;
        output<<"Discrepancy: "<<GPUError-CPUError<<endl;
    }
    cout<<"Performance gain: "<<CPUTime*1000/GPUTime<<" x"<<endl;
    output<<"Performance gain: "<<CPUTime*1000/GPUTime<<" x"<<endl;
};

void MCSimulator::Reader(string InputFile, MarketData &MarketInput, OptionDataContainer &OptionInput, GPUData &GPUInput, SimulationParameters &Parameters, bool &CPUComparison, ofstream& output){

    cout<<"Reading input file: "<<InputFile<<" ..."<<endl;
    fstream file;
    file.open(InputFile.c_str() , ios::in);

    if(file.fail()){
        cout<< "ERROR: input file not found! "<<  endl;
        output<< "ERROR: input file not found! "<<  endl;
        exit(1);
    }
    string temp, word;
    int Threads=0, Streams=0, BlockSize=0;
    int EulerApproximation=0, Antithetic=0;
    char OptionType[32], ProcessType[32];
    double Volatility=0, Drift=0;
    double InitialPrice=0, MaturityDate=0, StrikePrice=0;
    int DatesToSimulate=0, EulerSubStep=1;
    double ParamK=0, ParamB=0, ParamN=0;

    while (!file.eof()){
        file>>word;
        if (word=="THREADS") {
            file>> temp;
            Threads=atoi(temp.c_str());
        }
        if (word=="STREAMS"){
            file>> temp;
            Streams=atoi(temp.c_str());
        }
        if (word=="BLOCK_SIZE"){
            file>> temp;
            BlockSize=atoi(temp.c_str());
        }
        if (word=="EULER_APPROX"){
            file>> temp;
            EulerApproximation=atoi(temp.c_str());
        }
        if (word=="ANTITHETIC_VARIABLE"){
            file>> temp;
            Antithetic=atoi(temp.c_str());
        }
        if (word=="PROCESS_TYPE"){
            file>> temp;
            strcpy (ProcessType,temp.c_str());
        }
        if (word=="OPTION_TYPE"){
            file>> temp;
            strcpy (OptionType,temp.c_str());
        }
        if (word=="VOLATILITY"){
            file>> temp;
            Volatility=atof(temp.c_str());
        }
        if (word=="DRIFT"){
            file>> temp;
            Drift=atof(temp.c_str());
        }
        if (word=="INITIAL_PRICE"){
            file>> temp;
            InitialPrice=atof(temp.c_str());
        }
        if (word=="MATURITY_DATE"){
            file>> temp;
            MaturityDate=atof(temp.c_str());
        }
        if (word=="DATES_TO_SIMULATE"){
            file>> temp;
            DatesToSimulate=atoi(temp.c_str());
        }
        if (word=="STRIKE_PRICE"){
            file>> temp;
            StrikePrice=atof(temp.c_str());
        }
        if (word=="PARAMETER_K"){
            file>> temp;
            ParamK=atof(temp.c_str());
        }
        if (word=="PARAMETER_N"){
            file>> temp;
            ParamN=atof(temp.c_str());
        }
        if (word=="PARAMETER_B"){
            file>> temp;
            ParamB=atof(temp.c_str());
        }
        if (word=="EULER_SUB_STEP"){
            file>> temp;
            EulerSubStep=atof(temp.c_str());
        }
        if (word=="CPU_COMPARISON"){
            file>> temp;
            CPUComparison=atoi(temp.c_str());
        }
    }

    file.close();

    GPUInput.Threads=Threads;
    GPUInput.Streams=Streams;
    GPUInput.BlockSize=BlockSize;

    bool AntitheticVariable=Antithetic;
    bool EulerBool=EulerApproximation;
    Parameters.EulerApprox=EulerBool;
    Parameters.EulerSubStep=EulerSubStep;
    Parameters.AntitheticVariable=AntitheticVariable;

    MarketInput.Volatility=Volatility;
    MarketInput.Drift=Drift;
    MarketInput.EquityInitialPrice=InitialPrice;

    if(EulerBool==false)
        EulerSubStep=1;

    OptionInput.MaturityDate=MaturityDate;
    OptionInput.NumberOfFixingDate=DatesToSimulate,
    OptionInput.StrikePrice=StrikePrice;

    if(strcmp(ProcessType,"LOGNORMALSTD")==0)
        Parameters.ProcessType=0;

    if(strcmp(ProcessType,"LOGNORMALBIN")==0)
        Parameters.ProcessType=1;

    if(strcmp(OptionType,"FORWARD")==0)
        OptionInput.OptionType=0;

    if(strcmp(OptionType,"PLAIN_VANILLA_CALL")==0)
        OptionInput.OptionType=1;

    if(strcmp(OptionType,"PLAIN_VANILLA_PUT")==0)
        OptionInput.OptionType=2;

    if(strcmp(OptionType,"ABSOLUTE_PERFORMANCE_BARRIER")==0)
        OptionInput.OptionType=3;

    OptionInput.B=ParamB;
    OptionInput.N=ParamN;
    OptionInput.K=ParamK;

    output<<"Simulation parameters:"<<endl;
    output<<"Input file: "<<InputFile<<endl;
    output<<"Number of blocks requested: "<< (GPUInput.Threads + GPUInput.BlockSize - 1) / GPUInput.BlockSize<<endl;
    output<<"Thread per block requested: "<< GPUInput.BlockSize <<endl;
    output<<"CPU comparison: "<<CPUComparison<<endl;

};

//## Test di non-regressione ###################################################

int MCSimulator::RegressionTest(){

    //## Opzione di benchmark: Plain Vanilla Call coi seguenti parametri: ######

    MarketInput.Volatility=0.1;
    MarketInput.Drift=0.001;
    MarketInput.EquityInitialPrice=100.;

    OptionInput.MaturityDate=1.;
    OptionInput.NumberOfFixingDate=1;
    OptionInput.StrikePrice=100.;
    OptionInput.OptionType=1;

    GPUInput.Threads=5120;
    GPUInput.Streams=5000;
    GPUInput.BlockSize=512;

    Parameters.EulerApprox=0;
    Parameters.AntitheticVariable=0;
    Parameters.EulerSubStep=1;
    Parameters.ProcessType=0;

    sizeSeedVector = GPUInput.Threads * sizeof(Seed);
    sizeDevStVector = GPUInput.Threads * sizeof(Statistics);
    MemoryAllocationGPU(& PayOffsGPU,  & SeedVector, & _PayOffsGPU, & _SeedVector, sizeSeedVector, sizeDevStVector, GPUInput.Threads);

    GetSeeds(SeedVector, GPUInput.Threads, _Seed);
    cudaMemcpy(_SeedVector, SeedVector, sizeSeedVector, cudaMemcpyHostToDevice);

    cout<<"Simulating test..."<<endl;

    gridSize = (GPUInput.Threads + GPUInput.BlockSize - 1) / GPUInput.BlockSize;
    Kernel<<<gridSize, GPUInput.BlockSize>>>(_SeedVector, _PayOffsGPU, GPUInput.Streams, MarketInput, OptionInput, Parameters);

    cudaMemcpy(PayOffsGPU, _PayOffsGPU, sizeDevStVector, cudaMemcpyDeviceToHost);

    for (int i=0; i<GPUInput.Threads; i++)
        FinalStatisticsGPU=FinalStatisticsGPU+PayOffsGPU[i];

    if(FinalStatisticsGPU.GetMean()==4.0411858633024114)
        cout<<"PASSED!"<<endl;
    else{
        cout<<"FAILED!: "<<endl;
        cout<<"Expected price: "<<4.0411858633024114*exp(OptionInput.NumberOfFixingDate*MarketInput.Drift)<<endl;
        cout<<"Obtained price: "<<FinalStatisticsGPU.GetMean()*exp(OptionInput.NumberOfFixingDate*MarketInput.Drift)<<endl;

    }
    if(FinalStatisticsGPU.GetStDev()==0.0012322640294403595)
        cout<<"PASSED!"<<endl;
    else{
        cout<<"FAILED!: "<<endl;
        cout<<"Expected MC error: "<<0.0012322640294403595<<endl;
        cout<<"Obtained MC error: "<<FinalStatisticsGPU.GetStDev()<<endl;
    }

    MemoryDeallocationGPU(PayOffsGPU, SeedVector, _PayOffsGPU, _SeedVector);

    return 0;
};
