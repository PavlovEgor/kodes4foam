#include "mass_mole.cuh"
#include <stdio.h>
#include "mechanism.cuh"
#include "gpu_memory.cuh"
    //apply masking of ICs for cache optimized mechanisms
    void apply_mask(double* y_specs) {
        double temp [NSP];
        memcpy(temp, y_specs, NSP * sizeof(double));
        y_specs[0] = temp[0];
        y_specs[1] = temp[2];
        y_specs[2] = temp[3];
        y_specs[3] = temp[4];
        y_specs[4] = temp[5];
        y_specs[5] = temp[6];
        y_specs[6] = temp[7];
        y_specs[7] = temp[8];
        y_specs[8] = temp[9];
        y_specs[9] = temp[10];
        y_specs[10] = temp[11];
        y_specs[11] = temp[12];
        y_specs[12] = temp[13];
        y_specs[13] = temp[14];
        y_specs[14] = temp[15];
        y_specs[15] = temp[16];
        y_specs[16] = temp[17];
        y_specs[17] = temp[18];
        y_specs[18] = temp[19];
        y_specs[19] = temp[20];
        y_specs[20] = temp[21];
        y_specs[21] = temp[22];
        y_specs[22] = temp[23];
        y_specs[23] = temp[24];
        y_specs[24] = temp[25];
        y_specs[25] = temp[26];
        y_specs[26] = temp[27];
        y_specs[27] = temp[28];
        y_specs[28] = temp[29];
        y_specs[29] = temp[30];
        y_specs[30] = temp[31];
        y_specs[31] = temp[32];
        y_specs[32] = temp[33];
        y_specs[33] = temp[34];
        y_specs[34] = temp[35];
        y_specs[35] = temp[36];
        y_specs[36] = temp[37];
        y_specs[37] = temp[38];
        y_specs[38] = temp[39];
        y_specs[39] = temp[40];
        y_specs[40] = temp[41];
        y_specs[41] = temp[42];
        y_specs[42] = temp[43];
        y_specs[43] = temp[44];
        y_specs[44] = temp[45];
        y_specs[45] = temp[46];
        y_specs[46] = temp[47];
        y_specs[47] = temp[48];
        y_specs[48] = temp[49];
        y_specs[49] = temp[50];
        y_specs[50] = temp[51];
        y_specs[51] = temp[52];
        y_specs[52] = temp[53];
        y_specs[53] = temp[54];
        y_specs[54] = temp[55];
        y_specs[55] = temp[56];
        y_specs[56] = temp[57];
        y_specs[57] = temp[58];
        y_specs[58] = temp[59];
        y_specs[59] = temp[60];
        y_specs[60] = temp[61];
        y_specs[61] = temp[62];
        y_specs[62] = temp[63];
        y_specs[63] = temp[64];
        y_specs[64] = temp[65];
        y_specs[65] = temp[66];
        y_specs[66] = temp[67];
        y_specs[67] = temp[68];
        y_specs[68] = temp[69];
        y_specs[69] = temp[70];
        y_specs[70] = temp[71];
        y_specs[71] = temp[72];
        y_specs[72] = temp[73];
        y_specs[73] = temp[74];
        y_specs[74] = temp[75];
        y_specs[75] = temp[76];
        y_specs[76] = temp[77];
        y_specs[77] = temp[78];
        y_specs[78] = temp[79];
        y_specs[79] = temp[80];
        y_specs[80] = temp[81];
        y_specs[81] = temp[82];
        y_specs[82] = temp[83];
        y_specs[83] = temp[84];
        y_specs[84] = temp[85];
        y_specs[85] = temp[86];
        y_specs[86] = temp[87];
        y_specs[87] = temp[88];
        y_specs[88] = temp[89];
        y_specs[89] = temp[90];
        y_specs[90] = temp[91];
        y_specs[91] = temp[92];
        y_specs[92] = temp[93];
        y_specs[93] = temp[94];
        y_specs[94] = temp[95];
        y_specs[95] = temp[96];
        y_specs[96] = temp[97];
        y_specs[97] = temp[98];
        y_specs[98] = temp[99];
        y_specs[99] = temp[100];
        y_specs[100] = temp[101];
        y_specs[101] = temp[102];
        y_specs[102] = temp[103];
        y_specs[103] = temp[104];
        y_specs[104] = temp[105];
        y_specs[105] = temp[106];
        y_specs[106] = temp[107];
        y_specs[107] = temp[108];
        y_specs[108] = temp[109];
        y_specs[109] = temp[110];
        y_specs[110] = temp[111];
        y_specs[111] = temp[112];
        y_specs[112] = temp[113];
        y_specs[113] = temp[114];
        y_specs[114] = temp[115];
        y_specs[115] = temp[116];
        y_specs[116] = temp[117];
        y_specs[117] = temp[118];
        y_specs[118] = temp[119];
        y_specs[119] = temp[120];
        y_specs[120] = temp[121];
        y_specs[121] = temp[122];
        y_specs[122] = temp[123];
        y_specs[123] = temp[124];
        y_specs[124] = temp[125];
        y_specs[125] = temp[126];
        y_specs[126] = temp[127];
        y_specs[127] = temp[128];
        y_specs[128] = temp[129];
        y_specs[129] = temp[130];
        y_specs[130] = temp[131];
        y_specs[131] = temp[132];
        y_specs[132] = temp[133];
        y_specs[133] = temp[134];
        y_specs[134] = temp[135];
        y_specs[135] = temp[136];
        y_specs[136] = temp[137];
        y_specs[137] = temp[138];
        y_specs[138] = temp[139];
        y_specs[139] = temp[140];
        y_specs[140] = temp[141];
        y_specs[141] = temp[142];
        y_specs[142] = temp[143];
        y_specs[143] = temp[144];
        y_specs[144] = temp[145];
        y_specs[145] = temp[146];
        y_specs[146] = temp[147];
        y_specs[147] = temp[148];
        y_specs[148] = temp[149];
        y_specs[149] = temp[150];
        y_specs[150] = temp[151];
        y_specs[151] = temp[152];
        y_specs[152] = temp[153];
        y_specs[153] = temp[154];
        y_specs[154] = temp[155];
        y_specs[155] = temp[156];
        y_specs[156] = temp[157];
        y_specs[157] = temp[158];
        y_specs[158] = temp[1];
    }
    //reverse masking of ICs for cache optimized mechanisms
    void apply_reverse_mask(double* y_specs) {
        double temp [NSP];
        memcpy(temp, y_specs, NSP * sizeof(double));
        y_specs[0] = temp[0];
        y_specs[1] = temp[158];
        y_specs[2] = temp[1];
        y_specs[3] = temp[2];
        y_specs[4] = temp[3];
        y_specs[5] = temp[4];
        y_specs[6] = temp[5];
        y_specs[7] = temp[6];
        y_specs[8] = temp[7];
        y_specs[9] = temp[8];
        y_specs[10] = temp[9];
        y_specs[11] = temp[10];
        y_specs[12] = temp[11];
        y_specs[13] = temp[12];
        y_specs[14] = temp[13];
        y_specs[15] = temp[14];
        y_specs[16] = temp[15];
        y_specs[17] = temp[16];
        y_specs[18] = temp[17];
        y_specs[19] = temp[18];
        y_specs[20] = temp[19];
        y_specs[21] = temp[20];
        y_specs[22] = temp[21];
        y_specs[23] = temp[22];
        y_specs[24] = temp[23];
        y_specs[25] = temp[24];
        y_specs[26] = temp[25];
        y_specs[27] = temp[26];
        y_specs[28] = temp[27];
        y_specs[29] = temp[28];
        y_specs[30] = temp[29];
        y_specs[31] = temp[30];
        y_specs[32] = temp[31];
        y_specs[33] = temp[32];
        y_specs[34] = temp[33];
        y_specs[35] = temp[34];
        y_specs[36] = temp[35];
        y_specs[37] = temp[36];
        y_specs[38] = temp[37];
        y_specs[39] = temp[38];
        y_specs[40] = temp[39];
        y_specs[41] = temp[40];
        y_specs[42] = temp[41];
        y_specs[43] = temp[42];
        y_specs[44] = temp[43];
        y_specs[45] = temp[44];
        y_specs[46] = temp[45];
        y_specs[47] = temp[46];
        y_specs[48] = temp[47];
        y_specs[49] = temp[48];
        y_specs[50] = temp[49];
        y_specs[51] = temp[50];
        y_specs[52] = temp[51];
        y_specs[53] = temp[52];
        y_specs[54] = temp[53];
        y_specs[55] = temp[54];
        y_specs[56] = temp[55];
        y_specs[57] = temp[56];
        y_specs[58] = temp[57];
        y_specs[59] = temp[58];
        y_specs[60] = temp[59];
        y_specs[61] = temp[60];
        y_specs[62] = temp[61];
        y_specs[63] = temp[62];
        y_specs[64] = temp[63];
        y_specs[65] = temp[64];
        y_specs[66] = temp[65];
        y_specs[67] = temp[66];
        y_specs[68] = temp[67];
        y_specs[69] = temp[68];
        y_specs[70] = temp[69];
        y_specs[71] = temp[70];
        y_specs[72] = temp[71];
        y_specs[73] = temp[72];
        y_specs[74] = temp[73];
        y_specs[75] = temp[74];
        y_specs[76] = temp[75];
        y_specs[77] = temp[76];
        y_specs[78] = temp[77];
        y_specs[79] = temp[78];
        y_specs[80] = temp[79];
        y_specs[81] = temp[80];
        y_specs[82] = temp[81];
        y_specs[83] = temp[82];
        y_specs[84] = temp[83];
        y_specs[85] = temp[84];
        y_specs[86] = temp[85];
        y_specs[87] = temp[86];
        y_specs[88] = temp[87];
        y_specs[89] = temp[88];
        y_specs[90] = temp[89];
        y_specs[91] = temp[90];
        y_specs[92] = temp[91];
        y_specs[93] = temp[92];
        y_specs[94] = temp[93];
        y_specs[95] = temp[94];
        y_specs[96] = temp[95];
        y_specs[97] = temp[96];
        y_specs[98] = temp[97];
        y_specs[99] = temp[98];
        y_specs[100] = temp[99];
        y_specs[101] = temp[100];
        y_specs[102] = temp[101];
        y_specs[103] = temp[102];
        y_specs[104] = temp[103];
        y_specs[105] = temp[104];
        y_specs[106] = temp[105];
        y_specs[107] = temp[106];
        y_specs[108] = temp[107];
        y_specs[109] = temp[108];
        y_specs[110] = temp[109];
        y_specs[111] = temp[110];
        y_specs[112] = temp[111];
        y_specs[113] = temp[112];
        y_specs[114] = temp[113];
        y_specs[115] = temp[114];
        y_specs[116] = temp[115];
        y_specs[117] = temp[116];
        y_specs[118] = temp[117];
        y_specs[119] = temp[118];
        y_specs[120] = temp[119];
        y_specs[121] = temp[120];
        y_specs[122] = temp[121];
        y_specs[123] = temp[122];
        y_specs[124] = temp[123];
        y_specs[125] = temp[124];
        y_specs[126] = temp[125];
        y_specs[127] = temp[126];
        y_specs[128] = temp[127];
        y_specs[129] = temp[128];
        y_specs[130] = temp[129];
        y_specs[131] = temp[130];
        y_specs[132] = temp[131];
        y_specs[133] = temp[132];
        y_specs[134] = temp[133];
        y_specs[135] = temp[134];
        y_specs[136] = temp[135];
        y_specs[137] = temp[136];
        y_specs[138] = temp[137];
        y_specs[139] = temp[138];
        y_specs[140] = temp[139];
        y_specs[141] = temp[140];
        y_specs[142] = temp[141];
        y_specs[143] = temp[142];
        y_specs[144] = temp[143];
        y_specs[145] = temp[144];
        y_specs[146] = temp[145];
        y_specs[147] = temp[146];
        y_specs[148] = temp[147];
        y_specs[149] = temp[148];
        y_specs[150] = temp[149];
        y_specs[151] = temp[150];
        y_specs[152] = temp[151];
        y_specs[153] = temp[152];
        y_specs[154] = temp[153];
        y_specs[155] = temp[154];
        y_specs[156] = temp[155];
        y_specs[157] = temp[156];
        y_specs[158] = temp[157];
    }
void set_same_initial_conditions(int NUM, double** y_host, double** var_host) 
{
    double Xi [NSP] = {0.0};
    //set initial mole fractions here

    //Normalize mole fractions to sum to one
    double Xsum = 0.0;
    for (int j = 0; j < NSP; ++ j) {
        Xsum += Xi[j];
    }
    if (Xsum == 0.0) {
        printf("Use of the set initial conditions function requires user implementation!\n");
        exit(-1);
    }
    for (int j = 0; j < NSP; ++ j) {
        Xi[j] /= Xsum;
    }

    //convert to mass fractions
    double Yi[NSP - 1] = {0.0};
    mole2mass(Xi, Yi);

    //set initial pressure, units [PA]
    double P = 101325.0;
    // set intial temperature, units [K]
    double T0 = 1600;

    (*y_host) = (double*)malloc(NUM * NSP * sizeof(double));
    (*var_host) = (double*)malloc(NUM * sizeof(double));
    //load temperature and mass fractions for all threads (cells)
    for (int i = 0; i < NUM; ++i) {
        (*y_host)[i] = T0;
        //loop through species
        for (int j = 1; j < NSP; ++j) {
            (*y_host)[i + NUM * j] = Yi[j - 1];
        }
    }

#ifdef CONV
    //calculate density
    double rho = getDensity(T0, P, Xi);
#endif

    for (int i = 0; i < NUM; ++i) {
#ifdef CONV
        (*var_host)[i] = rho;
#elif defined(CONP)
        (*var_host)[i] = P;
#endif
    }
}

