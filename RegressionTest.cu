#define SEED 1889

// ## Non modificare SEED!######################################################

#include "MCSimulator.h"

int main(){

    //## implementazione: vedi file MCSimulator.cu #############################
    MCSimulator MCSim(SEED, "", "");
    return MCSim.RegressionTest();

}
