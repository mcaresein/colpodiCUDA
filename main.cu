/*##############################################################################################################################################################
# Pricer MonteCarlo di opzioni la cui dinamica e' determinata da processi lognormali esatti o approssimati.                                                    #
#                                                                                                                                                              #
# Usage: ./pricer                                                                                                                                              #
# Speficicare: Dati di input del processo (MarketData), Dati di input dell'opzione (OptionData), tipo di opzione (guarda in Option.h per quelle implementate), #
#             tipo di processo (guarda in StocasticProcess.h per quelli implementati). Vedi file input.conf in DATA                                            #
#                                                                                                                                                              #
# Output: Prezzo stimato secondo il Pay Off specificato e corrispondente errore MonteCarlo.                                                                    #
##############################################################################################################################################################*/

#define SEED 1995
#define INPUT_FILE "DATA/input.conf"
#define OUTPUT_FILE "DATA/output.dat"

#include "MCSimulator.h"

int main(){

    //## implementazione: vedi file MCSimulator.cu #############################
    MCSimulator MCSim(SEED, INPUT_FILE, OUTPUT_FILE);
    return MCSim.main();

}
