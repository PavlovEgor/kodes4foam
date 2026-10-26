#include <math.h>
#include "header.cuh"
#include "rates.cuh"

__device__ void get_rxn_pres_mod (const double T, const double pres, const double * __restrict__ C, double * __restrict__ pres_mod) {
  extern volatile __shared__ double shared_temp[];
  // third body variable declaration
  register double thd;

  // pressure dependence variable declarations
  register double k0;
  register double kinf;
  register double Pr;

  // troe variable declarations
  register double logFcent;
  register double A;
  register double B;

  register double logT = log(T);
  register double m = pres / (8.31446210e+03 * T);

  // reaction 0;
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(1)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(2)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(6)];
  shared_temp[threadIdx.x] = C[INDEX(10)];
  pres_mod[INDEX(0)] = m + 1.5 * shared_temp[threadIdx.x + 2 * blockDim.x] + 11.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.8999999999999999 * shared_temp[threadIdx.x] + 2.8 * C[INDEX(11)] - 0.17000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)];

  // reaction 3;
  pres_mod[INDEX(1)] = m + 1.5 * shared_temp[threadIdx.x + 2 * blockDim.x] + 11.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.17000000000000004 * C[INDEX(0)] + 0.8999999999999999 * shared_temp[threadIdx.x] + 2.8 * C[INDEX(11)] - 0.17000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)];

  // reaction 5;
  pres_mod[INDEX(2)] = m - 0.27 * shared_temp[threadIdx.x + 2 * blockDim.x] + 2.65 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.62 * C[INDEX(0)];

  // reaction 7;
  pres_mod[INDEX(3)] = m + 1.5 * shared_temp[threadIdx.x + 2 * blockDim.x] + 11.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.25 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] - 0.25 * shared_temp[threadIdx.x + 3 * blockDim.x] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)];

  // reaction 8;
  thd = m + 6.65 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.6000000000000001 * C[INDEX(11)] + 0.5 * C[INDEX(158)] + 0.19999999999999996 * C[INDEX(4)] - 0.35 * shared_temp[threadIdx.x + 3 * blockDim.x] + 6.7 * C[INDEX(8)] + 2.7 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.7999999999999998 * shared_temp[threadIdx.x];
  k0 = exp(4.9266569663351575e+01 - 2.3 * logT - (2.4531450567319323e+04 / T));
  kinf = exp(2.8324168296488494e+01 + 0.9 * logT - (2.4531450567319323e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.70000000e-01 * exp(-T / 1.00000000e-30) + 4.30000000e-01 * exp(-T / 1.00000000e+30), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(4)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 21;
  thd = m + 0.30000000000000004 * shared_temp[threadIdx.x + 2 * blockDim.x] + 0.8999999999999999 * shared_temp[threadIdx.x] + 2.8 * C[INDEX(11)] - 0.36 * shared_temp[threadIdx.x + 3 * blockDim.x] + 9.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.5 * C[INDEX(0)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)];
  k0 = exp(3.0487491322149033e+01 - 1.23 * logT);
  kinf = exp(2.2260133056545676e+01 + 0.44 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.30000000e-01 * exp(-T / 1.00000000e-30) + 6.70000000e-01 * exp(-T / 1.00000000e+30) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(5)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 22;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 11.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.75 * shared_temp[threadIdx.x] + 2.6 * C[INDEX(11)] - 0.30000000000000004 * C[INDEX(0)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(4.1603535422702485e+01 - 2.79 * logT - (2.1089931963247509e+03 / T));
  kinf = exp(1.6427049858685642e+01 - (1.1996754426242439e+03 / T));
  Pr = k0 * thd / kinf;
  pres_mod[INDEX(6)] =  Pr / (1.0 + Pr);

  // reaction 30;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(6.3078056071016121e+01 - 4.76 * logT - (1.2278557382563572e+03 / T));
  kinf = exp(3.0172623109393093e+01 - 0.63 * logT - (1.9273309334105934e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.17000000e-01 * exp(-T / 7.40000000e+01) + 7.83000000e-01 * exp(-T / 2.94100000e+03) + exp(-6.96400000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(7)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 51;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(4.9517437762680643e+01 - 3.14 * logT - (6.1896006477677020e+02 / T));
  kinf = exp(3.0849896940796750e+01 - 0.8 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.20000000e-01 * exp(-T / 7.80000000e+01) + 6.80000000e-01 * exp(-T / 1.99500000e+03) + exp(-5.59000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(8)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 65;
  thd = m;
  k0 = exp(4.3370780326166958e+01 - 3.0 * logT);
  kinf = exp(1.5871171570978706e+01 + 0.9 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.00000000e-01 * exp(-T / 1.00000000e+03) + 6.00000000e-01 * exp(-T / 7.00000000e+01) + exp(-1.70000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(9)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 85;
  thd = m;
  k0 = exp(9.2508868827869989e+01 - 6.995 * logT - (4.9311592243592044e+04 / T));
  kinf = exp(4.2180820797783944e+01 - 0.615 * logT - (4.6568240463805836e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.47480000e+00 * exp(-T / 3.55800000e+04) + -4.74800000e-01 * exp(-T / 1.11600000e+03) + exp(-9.02300000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(10)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 86;
  thd = m;
  k0 = exp(1.0167141853600982e+02 - 8.227 * logT - (5.0028629801559873e+04 / T));
  kinf = exp(4.2584685137181467e+01 - 1.017 * logT - (4.6151272732363534e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(-1.54500000e+00 * exp(-T / 3.29000000e+03) + 2.54500000e+00 * exp(-T / 4.73200000e+04) + exp(-4.71100000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(11)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 87;
  thd = m;
  k0 = exp(9.1021648548160144e+01 - 7.244 * logT - (5.2953945775999156e+04 / T));
  kinf = exp(-4.8413989768509564e+00 + 5.038 * logT - (4.2505648272784842e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(7.49100000e+01 * exp(-T / 3.70500000e+04) + -7.39100000e+01 * exp(-T / 4.15000000e+04) + exp(-5.22000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(12)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 118;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(4.1746636266343160e+01 - 2.57 * logT - (7.1708787992430689e+02 / T));
  kinf = exp(2.0809443533187462e+01 + 0.48 * logT - (-1.3083708686338232e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.17600000e-01 * exp(-T / 2.71000000e+02) + 7.82400000e-01 * exp(-T / 2.75500000e+03) + exp(-6.57000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(13)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 119;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(4.9977627770478051e+01 - 3.42 * logT - (4.2445563856740657e+04 / T));
  kinf = exp(1.0668955394675699e+01 + 1.5 * logT - (4.0056277362789355e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(6.80000000e-02 * exp(-T / 1.97000000e+02) + 9.32000000e-01 * exp(-T / 1.54000000e+03) + exp(-1.03000000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(14)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 129;
  pres_mod[INDEX(15)] = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)];

  // reaction 139;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)];
  k0 = exp(6.0106229318315691e+01 - 4.82 * logT - (3.2860237585303325e+03 / T));
  kinf = exp(2.0107079697522593e+01 + 0.454 * logT - (1.8115904334929860e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.81300000e-01 * exp(-T / 1.03000000e+02) + 7.18700000e-01 * exp(-T / 1.29100000e+03) + exp(-4.16000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(16)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 140;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)];
  k0 = exp(5.1282810476735499e+01 - 3.0 * logT - (1.2231757963031669e+04 / T));
  kinf = exp(3.1850528821104653e+01 - (1.3169256012364289e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.00000000e-01 * exp(-T / 2.50000000e+03) + 9.00000000e-01 * exp(-T / 1.30000000e+03) + exp(-1.00000000e+99 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(17)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 153;
  thd = m + 4.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * shared_temp[threadIdx.x] + 2.0 * C[INDEX(11)];
  k0 = exp(5.9650299416281612e+01 - 3.75 * logT - (4.9396032486575416e+02 / T));
  kinf = exp(2.8453879903010151e+01 - 0.69 * logT - (8.8013101893867571e+01 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.00000000e+00 * exp(-T / 5.70000000e+02) + 0.00000000e+00 * exp(-T / 1.00000000e+30) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(18)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 154;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(8.1278612893528006e+01 - 7.08 * logT - (3.3640227910835029e+03 / T));
  kinf = exp(3.3886771157681913e+01 - 0.99 * logT - (7.9508691247747720e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.58000000e-01 * exp(-T / 1.25000000e+02) + 8.42000000e-01 * exp(-T / 2.21900000e+03) + exp(-6.88200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(19)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 166;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * C[INDEX(0)];
  k0 = exp(7.6335964940416673e+01 - 6.642 * logT - (2.9030736696725098e+03 / T));
  kinf = exp(1.3771454171767354e+01 + 1.463 * logT - (6.8186251038416549e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.56900000e+00 * exp(-T / 2.99000000e+02) + -5.69000000e-01 * exp(T / 9.14700000e+03) + exp(-1.52400000e+02 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(20)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 203;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(5.5598514468478307e+01 - 3.86 * logT - (1.6706889553324204e+03 / T));
  kinf = exp(2.2528270532924488e+01 + 0.27 * logT - (1.4090147816056557e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.18000000e-01 * exp(-T / 2.07500000e+02) + 7.82000000e-01 * exp(-T / 2.66300000e+03) + exp(-6.09500000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(21)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 204;
  pres_mod[INDEX(22)] = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * C[INDEX(0)];

  // reaction 223;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(5.9413082137755744e+01 - 4.664 * logT - (1.9021699551676352e+03 / T));
  kinf = exp(1.6654589021472887e+01 + 1.266 * logT - (1.3632218012034718e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.12000000e-01 * exp(T / 1.02000000e+04) + 7.88000000e-01 * exp(-T / 1.00000000e-30), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(23)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 242;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(6.3491553350821555e+01 - 4.8 * logT - (9.5611717323240919e+02 / T));
  kinf = exp(3.2236191301916641e+01 - 1.0 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.54000000e-01 * exp(-T / 1.32000000e+02) + 6.46000000e-01 * exp(-T / 1.31500000e+03) + exp(-5.56600000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(24)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 257;
  thd = m;
  k0 = exp(1.2897432400990812e+02 - 11.3 * logT - (4.8265046514554444e+04 / T));
  kinf = exp(5.1552960070425641e+01 - 1.74 * logT - (4.3455525523413002e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(9.97510000e-01 * exp(-T / 7.18100000e+02) + 2.49000000e-03 * exp(-T / 6.08900000e+00) + exp(-3.78000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(25)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 258;
  thd = m;
  k0 = exp(1.2677320837707892e+02 - 11.3 * logT - (4.8265046514554444e+04 / T));
  kinf = exp(4.9354918833182865e+01 - 1.74 * logT - (4.3455525523413002e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(9.97510000e-01 * exp(-T / 7.18100000e+02) + 2.49000000e-03 * exp(-T / 6.08900000e+00) + exp(-3.78000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(26)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 270;
  thd = m;
  k0 = exp(1.1094279479857238e+02 - 10.27 * logT - (2.7873331697549023e+04 / T));
  kinf = exp(1.8360187363523401e+01 + 1.917 * logT - (2.2638439208232125e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.99100000e-01 * exp(-T / 8.10300000e+09) + 6.00900000e-01 * exp(-T / 6.67700000e+02) + exp(-5.00000000e+09 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(27)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 274;
  thd = m;
  k0 = exp(6.1658971887073243e+01 - 3.8 * logT - (2.1851756062487799e+04 / T));
  kinf = exp(3.4896450839182499e+01 - 0.15 * logT - (2.2946812157577820e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.50000000e-02 * exp(-T / 3.93000000e+02) + 9.85000000e-01 * exp(-T / 9.80000000e+09) + exp(-5.00000000e+09 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(28)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 275;
  thd = m;
  k0 = exp(7.1330947638624650e+01 - 5.07 * logT - (2.0782968028683423e+04 / T));
  kinf = exp(2.8706023538957524e+01 + 0.29 * logT - (2.0279748463824260e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.00000000e+00 * exp(-T / 1.15000000e+03) + 7.13000000e-17 * exp(-T / 4.99000000e+09) + exp(-1.79000000e+09 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(29)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 276;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(2.8729633404596658e+01 - (2.8683515196972276e+04 / T));
  kinf = exp(3.1032218497590705e+01 - (3.5728589105000552e+04 / T));
  Pr = k0 * thd / kinf;
  pres_mod[INDEX(30)] =  Pr / (1.0 + Pr);

  // reaction 287;
  pres_mod[INDEX(31)] = m;

  // reaction 324;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 0.30000000000000004 * C[INDEX(0)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 1.0 * C[INDEX(13)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * shared_temp[threadIdx.x + 3 * blockDim.x];
  k0 = exp(1.6521342566808721e+02 - 15.74 * logT - (4.9674816125507394e+04 / T));
  kinf = exp(8.5450290659153268e+01 - 5.84 * logT - (4.9003521225985271e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(6.90000000e-01 * exp(-T / 5.00000000e+01) + 3.10000000e-01 * exp(-T / 3.00000000e+03) + exp(-9.00000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(32)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 360;
  thd = m;
  k0 = exp(1.2118603866293091e+02 - 11.94 * logT - (4.9163545047610478e+03 / T));
  kinf = 25000000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(8.25000000e-01 * exp(-T / 1.34100000e+03) + 1.75000000e-01 * exp(-T / 6.00000000e+04) + exp(-1.01400000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(33)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 580;
  thd = m;
  k0 = exp(3.6090585194508151e+01 - (2.4948619586587571e+04 / T));
  kinf = exp(8.5499449895111354e+01 - 6.036 * logT - (4.6763690942797133e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(9.20020000e-01 * exp(-T / 1.00000000e-20) + 7.99800000e-02 * exp(-T / 3.24300000e+04) + exp(-4.85800000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(34)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 581;
  thd = m;
  k0 = exp(3.3911416954888743e+01 - (2.1617809286784774e+04 / T));
  kinf = exp(1.2162149448472276e+02 - 10.626 * logT - (5.0488018942319795e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(9.04980000e-01 * exp(-T / 1.00000000e-20) + 9.50200000e-02 * exp(-T / 5.34800000e+03) + exp(-4.32600000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(35)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 592;
  thd = m;
  k0 = exp(3.7720988235407297e+01 - (2.6457271842035341e+04 / T));
  kinf = exp(7.2304396784338749e+01 - 4.102 * logT - (4.6042074086789093e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(6.33800000e-01 * exp(-T / 8.15300000e+02) + 3.66200000e-01 * exp(-T / 6.07900000e+01) + exp(-1.00000000e+20 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(36)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 642;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * C[INDEX(0)];
  k0 = exp(1.2570313239567574e+02 - 12.81 * logT - (3.1451222803697674e+03 / T));
  kinf = exp(2.7120195492162559e+01 - 0.32 * logT - (-1.3199449186255839e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(8.96000000e-01 * exp(-T / 1.60600000e+03) + 1.04000000e-01 * exp(-T / 6.00000000e+04) + exp(-6.11800000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(37)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 643;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * C[INDEX(0)];
  k0 = exp(1.1556750958063344e+02 - 11.79 * logT - (4.5211761804771477e+03 / T));
  kinf = 15000000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(8.02000000e-01 * exp(-T / 2.27800000e+03) + 1.98000000e-01 * exp(-T / 6.00000000e+04) + exp(-5.72300000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(38)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 644;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * C[INDEX(0)];
  k0 = exp(9.7810513984510706e+01 - 9.32 * logT - (2.9355816535624122e+03 / T));
  kinf = 36000000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.02000000e-01 * exp(-T / 1.31400000e+03) + 4.98000000e-01 * exp(-T / 1.31400000e+03) + exp(-5.00000000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(39)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 646;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * C[INDEX(0)];
  k0 = exp(1.2462477396391213e+02 - 12.0 * logT - (3.0031137191665116e+03 / T));
  kinf = 50000000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(9.80000000e-01 * exp(-T / 1.09700000e+03) + 2.00000000e-02 * exp(-T / 1.09700000e+04) + exp(-6.86000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(40)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 647;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * C[INDEX(0)] + 2.0 * C[INDEX(35)] + 2.0 * C[INDEX(33)];
  k0 = exp(1.2187918584349086e+02 - 11.94 * logT - (4.9163545047610478e+03 / T));
  kinf = 50000000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(8.25000000e-01 * exp(-T / 1.34100000e+03) + 1.75000000e-01 * exp(-T / 6.00000000e+04) + exp(-1.01400000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(41)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 723;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)];
  k0 = exp(1.6076544106562153e+02 - 16.3 * logT - (3.5225369540141392e+03 / T));
  kinf = exp(2.5307212386482878e+01 + 0.15 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(0.00000000e+00 * exp(-T / 1.00000000e-01) + 1.00000000e+00 * exp(-T / 5.84900000e+02) + exp(-6.11300000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(42)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 773;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 2.0 * C[INDEX(29)] + 1.5 * C[INDEX(35)] + 1.5 * C[INDEX(33)];
  k0 = exp(5.7779738704468087e+01 - 4.718 * logT - (9.4152380585149353e+02 / T));
  kinf = exp(1.8234351165760874e+01 + 0.899 * logT - (-1.8266870204387607e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(0.00000000e+00 * exp(-T / 1.00000000e+02) + 1.00000000e+00 * exp(-T / 5.61300000e+03) + exp(-1.33900000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(43)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 830;
  pres_mod[INDEX(44)] = m;

  // reaction 967;
  thd = m;
  k0 = exp(1.8756370456228922e+02 - 18.6 * logT - (5.0855369224666989e+04 / T));
  kinf = exp(6.0129576682312681e+01 - 3.0 * logT - (4.3481139399264328e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(9.97509000e-01 * exp(-T / 3.76800000e+02) + 2.49100000e-03 * exp(-T / 6.08900000e+00) + exp(-4.63200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(45)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 968;
  thd = m;
  k0 = exp(1.9364825953246321e+02 - 19.4 * logT - (5.0966077528936003e+04 / T));
  kinf = exp(5.7713047329969413e+01 - 2.8 * logT - (4.3135075304510683e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(9.97509000e-01 * exp(-T / 3.72500000e+02) + 2.49100000e-03 * exp(-T / 6.08900000e+00) + exp(-5.25200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(46)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 1006;
  pres_mod[INDEX(47)] = m;

  // reaction 1117;
  thd = m + 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 1.0 * C[INDEX(13)] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(11)] + 2.0 * C[INDEX(29)] - 0.30000000000000004 * C[INDEX(0)];
  k0 = exp(1.2462477396391213e+02 - 12.0 * logT - (3.0031137191665116e+03 / T));
  kinf = 100000000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(9.80000000e-01 * exp(-T / 1.09700000e+03) + 2.00000000e-02 * exp(-T / 1.09700000e+04) + exp(-6.86000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(48)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 1999;
  thd = m;
  k0 = exp(3.0722063569286863e+01 - (4.7031152141519779e+04 / T));
  kinf = exp(4.0173565998079937e+01 - 0.39 * logT - (5.5519308795694677e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.80000000e-01 * exp(-T / 4.58100000e+03) + 4.20000000e-01 * exp(-T / 1.02000000e+02) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(49)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2000;
  thd = m;
  k0 = exp(2.7033184115172929e+01 - (4.6874902466631007e+04 / T));
  kinf = exp(3.6618217936590518e+01 - 0.39 * logT - (5.5519308795694677e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.80000000e-01 * exp(-T / 4.58100000e+03) + 4.20000000e-01 * exp(-T / 1.02000000e+02) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(50)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2023;
  thd = m;
  k0 = exp(7.9974292115367788e+01 - 5.96 * logT - (3.3606512199989469e+04 / T));
  kinf = exp(4.6388174096502127e+01 - 1.31 * logT - (3.2246309716175150e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(6.90000000e-01 * exp(-T / 1.00000000e-30) + 3.10000000e-01 * exp(-T / 1.00000000e+30) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(51)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2036;
  pres_mod[INDEX(52)] = m;

  // reaction 2138;
  thd = m + 0.7 * C[INDEX(158)] + 0.5 * C[INDEX(4)] + 9.0 * shared_temp[threadIdx.x + 1 * blockDim.x];
  k0 = exp(4.2998340473490288e+01 - 2.87 * logT - (7.7999032553170230e+02 / T));
  kinf = exp(2.7893385380396040e+01 - 0.75 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.50000000e-01 * exp(-T / 1.00000000e+03) + 7.50000000e-01 * exp(-T / 1.00000000e+05) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(53)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2141;
  thd = m + 0.6000000000000001 * C[INDEX(158)];
  k0 = exp(1.9296149481306266e+01 + 0.206 * logT - (-7.7999032553170230e+02 / T));
  kinf = exp(2.8036486224036711e+01 - 0.41 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.80000000e-01 * exp(-T / 1.00000000e-30) + 8.20000000e-01 * exp(-T / 1.00000000e+30) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(54)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2178;
  pres_mod[INDEX(55)] = m + 9.0 * shared_temp[threadIdx.x + 1 * blockDim.x];

  // reaction 2199;
  thd = m;
  k0 = exp(3.3152482033790797e+01 - 1.5 * logT);
  kinf = exp(2.1976028805441779e+01 + 0.24 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.90000000e-01 * exp(-T / 1.00000000e-30) + 7.10000000e-01 * exp(-T / 1.70000000e+03) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(56)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2220;
  pres_mod[INDEX(57)] = m;

  // reaction 2252;
  pres_mod[INDEX(58)] = m;

  // reaction 2393;
  thd = m + 0.7 * C[INDEX(158)] + 0.3999999999999999 * C[INDEX(4)] + 2.0 * C[INDEX(11)] + 11.0 * shared_temp[threadIdx.x + 1 * blockDim.x];
  k0 = exp(2.7123523282255231e+01 - (2.8906944683769743e+04 / T));
  kinf = exp(2.5318385687081001e+01 - (2.9136916024910381e+04 / T));
  Pr = k0 * thd / kinf;
  pres_mod[INDEX(59)] =  Pr / (1.0 + Pr);

  // reaction 2411;
  thd = m;
  k0 = exp(3.2467303022880024e+01 - (2.1135221724084837e+04 / T));
  kinf = exp(3.7417974852208722e+01 - (2.9438344544261021e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(8.17000000e-01 * exp(-T / 1.00000000e-30) + 1.83000000e-01 * exp(-T / 1.00000000e+30), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(60)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2419;
  thd = m;
  k0 = exp(4.4827526348237456e+01 - 3.0 * logT);
  kinf = 30000000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(6.00000000e-01 * exp(-T / 1.00000000e-30) + 4.00000000e-01 * exp(-T / 1.00000000e+30) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(61)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2421;
  thd = m;
  k0 = exp(4.5845906946901316e+01 - 2.8 * logT);
  kinf = exp(2.7120195492162559e+01 - 0.6 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(0.00000000e+00 * exp(-T / 1.00000000e-30) + 1.00000000e+00 * exp(-T / 9.00000000e+02) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(62)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2423;
  thd = m;
  k0 = exp(2.3942141661814613e+01 - (-1.4296467837648813e+03 / T));
  kinf = exp(2.2920490414282632e+01 - (9.6618156452959255e+01 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(-4.00000000e+00 * exp(-T / 1.00000000e-30) + 5.00000000e+00 * exp(-T / 1.20000000e+02) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(63)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2424;
  thd = m;
  k0 = exp(4.0212099662082174e+01 - 1.74 * logT);
  kinf = exp(2.8419478476292820e+01 - 0.88 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.00000000e-01 * exp(-T / 1.00000000e-30) + 6.00000000e-01 * exp(-T / 1.00000000e+30) + exp(-1.00000000e+30 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(64)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 2448;
  pres_mod[INDEX(65)] = m;

  // reaction 2456;
  pres_mod[INDEX(66)] = m;

} // end get_rxn_pres_mod

