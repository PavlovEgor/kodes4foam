#include "jacob.cuh"

#if defined(CONP)

__device__ void eval_jacob (const double t, const double pres, const double * __restrict__ y, double * __restrict__ jac, const mechanism_memory * __restrict__ d_mem) {

  extern volatile __shared__ double shared_temp[];
  double T = y[INDEX(0)];

  // average molecular weight
  double mw_avg;
  // mass-averaged density
  double rho;
  // species molar concentrations
  double * __restrict__ conc = d_mem->conc;
  double y_N;
  eval_conc(y[INDEX(0)], pres, &y[GRID_DIM], &y_N, &mw_avg, &rho, conc);

  // evaluate reaction rates
  double * __restrict__ fwd_rates = d_mem->fwd_rates;
  double * __restrict__ rev_rates = d_mem->rev_rates;
  eval_rxn_rates (T, pres, conc, fwd_rates, rev_rates);

  double * __restrict__ pres_mod = d_mem->pres_mod;
  //get pressure modifications to reaction rates
  get_rxn_pres_mod (T, pres, conc, pres_mod);

  // evaluate rate of change of species molar concentration
  double * __restrict__ spec_rates = d_mem->spec_rates;
  eval_spec_rates (fwd_rates, rev_rates, pres_mod, spec_rates, &spec_rates[INDEX(9)]);

  register double m = pres / (8.31446210e+03 * T);
  register double conc_temp;
  register double logT = log(T);
  register double j_temp = 0.0;
  register double kf = 0.0;
  register double pres_mod_temp = 0.0;
  register double Kc = 0.0;
  register double kr = 0;
  register double Pr = 0.0;
  double Fcent = 0.0;
  double A = 0.0;
  double B = 0.0;
  double lnF_AB = 0.0;
  register double rho_inv = 1.0 / rho;
  double * __restrict__ dBdT = d_mem->dBdT;
  if (T <= 1000.0) {
    dBdT[INDEX(0)] = (1.3443311200000001e+00 + -9.1793517299999996e+02 / T) / T + 3.9902603749999996e-03 + T * (-6.4927169999999995e-06 + T * (5.0393023500000001e-09 + -1.4752235220000002e-12 * T));
    dBdT[INDEX(1)] = (1.5000000000000000e+00 + 2.5473659899999999e+04 / T) / T + 3.5266640949999999e-13 + T * (-6.6530654666666665e-16 + T * (5.7520407999999997e-19 + -1.8554646640000001e-22 * T));
    dBdT[INDEX(2)] = (2.1682671000000000e+00 + 2.9122259200000000e+04 / T) / T + -1.6396594200000000e-03 + T * (2.2143546533333334e-06 + T * (-1.5320165600000000e-09 + 4.2253194199999998e-13 * T));
    dBdT[INDEX(3)] = (2.7824563599999999e+00 + -1.0639435599999999e+03 / T) / T + -1.4983670799999999e-03 + T * (3.2824340033333332e-06 + T * (-2.4203237725000002e-09 + 6.4874567400000000e-13 * T));
    dBdT[INDEX(4)] = (2.9920154299999999e+00 + 3.6150805599999999e+03 / T) / T + -1.2006587600000001e-03 + T * (1.5393128033333333e-06 + T * (-9.7028333249999999e-10 + 2.7282293999999999e-13 * T));
    dBdT[INDEX(5)] = (3.1986405600000003e+00 + -3.0293726699999999e+04 / T) / T + -1.0182170500000001e-03 + T * (2.1734673700000000e-06 + T * (-1.3719926550000001e-09 + 3.5439563400000000e-13 * T));
    dBdT[INDEX(6)] = (3.3017980099999997e+00 + 2.9480804000000001e+02 / T) / T + -2.3745602549999998e-03 + T * (7.0527630333333326e-06 + T * (-6.0690973500000000e-09 + 1.8584502480000002e-12 * T));
    dBdT[INDEX(7)] = (3.2761126899999997e+00 + -1.7702582100000000e+04 / T) / T + -2.7141120849999997e-04 + T * (5.5778567000000005e-06 + T * (-5.3942703249999999e-09 + 1.7249087260000000e-12 * T));
    dBdT[INDEX(8)] = (1.5000000000000000e+00 + -7.4537500000000000e+02 / T) / T + 0.0000000000000000e+00 + T * (0.0000000000000000e+00 + T * (0.0000000000000000e+00 + 0.0000000000000000e+00 * T));
    dBdT[INDEX(9)] = (2.2986770000000001e+00 + -1.0208999000000000e+03 / T) / T + 7.0412019999999999e-04 + T * (-1.3210739999999999e-06 + T * (1.4103787500000001e-09 + -4.8897079999999994e-13 * T));
  } else {
    dBdT[INDEX(0)] = (2.3372791999999998e+00 + -9.5015892199999996e+02 / T) / T + -2.4701236549999999e-05 + T * (1.6648559266666665e-07 + T * (-4.4891598500000001e-11 + 4.0051075199999998e-15 * T));
    dBdT[INDEX(1)] = (1.5000000099999999e+00 + 2.5473659899999999e+04 / T) / T + -1.1542148649999999e-11 + T * (5.3853982666666673e-15 + T * (-1.1837880875000001e-18 + 9.9639471400000006e-23 * T));
    dBdT[INDEX(2)] = (1.5694207800000002e+00 + 2.9217579099999999e+04 / T) / T + -4.2987056850000002e-05 + T * (1.3982819633333334e-08 + T * (-2.5044449749999998e-12 + 2.4566738199999997e-16 * T));
    dBdT[INDEX(3)] = (2.2825378399999998e+00 + -1.0884577200000001e+03 / T) / T + 7.4154376999999996e-04 + T * (-2.5265555633333331e-07 + T * (5.2367638749999998e-11 + -4.3343558799999996e-15 * T));
    dBdT[INDEX(4)] = (2.0928876700000001e+00 + 3.8586570000000002e+03 / T) / T + 2.7421485799999999e-04 + T * (4.2168409333333331e-08 + T * (-2.1986538899999999e-11 + 2.3482475199999999e-15 * T));
    dBdT[INDEX(5)] = (2.0339924900000002e+00 + -3.0004297100000000e+04 / T) / T + 1.0884590200000001e-03 + T * (-5.4690839333333327e-08 + T * (-2.4260496750000000e-11 + 3.3640198399999998e-15 * T));
    dBdT[INDEX(6)] = (3.0172109000000003e+00 + 1.1185671300000000e+02 / T) / T + 1.1199100650000000e-03 + T * (-2.1121938333333332e-07 + T * (2.8561592500000001e-11 + -2.1581707000000000e-15 * T));
    dBdT[INDEX(7)] = (3.1650028499999996e+00 + -1.7861787700000001e+04 / T) / T + 2.4541584700000000e-03 + T * (-6.3379741666666662e-07 + T * (9.2796496499999996e-11 + -5.7581661000000002e-15 * T));
    dBdT[INDEX(8)] = (1.5000000000000000e+00 + -7.4537500000000000e+02 / T) / T + 0.0000000000000000e+00 + T * (0.0000000000000000e+00 + T * (0.0000000000000000e+00 + 0.0000000000000000e+00 * T));
    dBdT[INDEX(9)] = (1.9266399999999999e+00 + -9.2279769999999996e+02 / T) / T + 7.4398840000000003e-04 + T * (-1.8949200000000001e-07 + T * (2.5242595000000000e-11 + -1.3506701999999999e-15 * T));
  }

  shared_temp[threadIdx.x + 3 * blockDim.x] = pres_mod[INDEX(0)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = conc[INDEX(2)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = fwd_rates[INDEX(0)];
  shared_temp[threadIdx.x] = rev_rates[INDEX(0)];
  //partial of rxn 0 wrt T
  j_temp = ((-shared_temp[threadIdx.x + 3 * blockDim.x] * (shared_temp[threadIdx.x + 1 * blockDim.x] - shared_temp[threadIdx.x]) / T) + (shared_temp[threadIdx.x + 3 * blockDim.x] / T) * ((shared_temp[threadIdx.x + 1 * blockDim.x] - shared_temp[threadIdx.x]) * (-1.0000000000000000e+00) + shared_temp[threadIdx.x + 1 * blockDim.x] * -1.0 - shared_temp[threadIdx.x] * (-T * (dBdT[INDEX(3)] - 2.0 * dBdT[INDEX(2)])))) * rho_inv;
  jac[INDEX(3)] = j_temp * -2.0 * 1.5999400000000000e+01;
  jac[INDEX(4)] = j_temp * 3.1998799999999999e+01;

  //partial of rxn 0 wrt species
  if (T <= 1000.0) {
    Kc = (2.1078866500000002e+00 + -1.5540778400000002e+00 * logT + T * (1.7809517600000000e-03 + T * (-5.7313765166666677e-07 + T * (2.1456978249999992e-10 + -4.9079552499999993e-14 * T))) - -5.9308461960000001e+04 / T);
  } else {
    Kc = (-2.2591422699999990e+00 + -8.5630372000000055e-01 * logT + T * (8.2751788369999995e-04 + T * (-1.4031059780000000e-07 + T * (1.9125509566666665e-11 + -1.2064226609999998e-15 * T))) - -5.9523615919999997e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  pres_mod_temp = (shared_temp[threadIdx.x + 1 * blockDim.x] - shared_temp[threadIdx.x]);
  j_temp = -mw_avg * rho_inv * shared_temp[threadIdx.x + 3 * blockDim.x] * (3.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 2.0 * shared_temp[threadIdx.x]);
  kf = exp(2.5510757579728459e+01 - 1.0 * logT);
  kr = kf / Kc;
  jac[INDEX(13)] =  -1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01 + 2.3280389298294963e+00 * pres_mod_temp);
  jac[INDEX(14)] =  1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01 + 2.3280389298294963e+00 * pres_mod_temp);
  jac[INDEX(23)] =  -3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp);
  jac[INDEX(24)] =  3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp);
  jac[INDEX(33)] =  -2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(34)] =  2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(43)] =  -(j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (-kr));
  jac[INDEX(44)] =  (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (-kr));
  jac[INDEX(53)] =  -1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp);
  jac[INDEX(54)] =  1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp);
  jac[INDEX(63)] =  -1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01 + 1.4756906746323557e+01 * pres_mod_temp);
  jac[INDEX(64)] =  1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01 + 1.4756906746323557e+01 * pres_mod_temp);
  jac[INDEX(73)] =  -9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(74)] =  9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(83)] =  -9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(84)] =  9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(93)] =  -8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01 + -5.9602775520927775e-01 * pres_mod_temp);
  jac[INDEX(94)] =  8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01 + -5.9602775520927775e-01 * pres_mod_temp);

  shared_temp[threadIdx.x + 3 * blockDim.x] = pres_mod[INDEX(1)];
  shared_temp[threadIdx.x] = fwd_rates[INDEX(1)];
  //partial of rxn 1 wrt T
  j_temp = ((-shared_temp[threadIdx.x + 3 * blockDim.x] * (shared_temp[threadIdx.x] - rev_rates[INDEX(1)]) / T) + (shared_temp[threadIdx.x + 3 * blockDim.x] / T) * ((shared_temp[threadIdx.x] - rev_rates[INDEX(1)]) * (-1.0000000000000000e+00) + shared_temp[threadIdx.x] * -1.0 - rev_rates[INDEX(1)] * (-T * (dBdT[INDEX(4)] - dBdT[INDEX(1)] - dBdT[INDEX(2)])))) * rho_inv;
  jac[INDEX(2)] = -j_temp * 1.0079400000000001e+00;
  jac[INDEX(3)] += -j_temp * 1.5999400000000000e+01;
  jac[INDEX(5)] = j_temp * 1.7007339999999999e+01;

  //partial of rxn 1 wrt species
  if (T <= 1000.0) {
    Kc = (-3.2924395000000217e-02 + -6.7625167000000008e-01 * logT + T * (4.3900065964733344e-04 + T * (-3.3752092466734670e-07 + T * (1.8724440897493195e-10 + -3.7427250453613386e-14 * T))) - -5.0980838539999997e+04 / T);
  } else {
    Kc = (2.1155734940000004e+00 + -9.7653312000000003e-01 * logT + T * (3.1720192639214862e-04 + T * (1.4092792157300865e-08 + T * (-6.4940309137373039e-12 + 5.2564500959013211e-16 * T))) - -5.0832582000000002e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  pres_mod_temp = (shared_temp[threadIdx.x] - rev_rates[INDEX(1)]);
  j_temp = -mw_avg * rho_inv * shared_temp[threadIdx.x + 3 * blockDim.x] * (3.0 * shared_temp[threadIdx.x] - 2.0 * rev_rates[INDEX(1)]);
  kf = exp(2.6937873935368604e+01 - 1.0 * logT);
  kr = kf / Kc;
  jac[INDEX(12)] =  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01 + 1.9280389298294964e+00 * pres_mod_temp);
  jac[INDEX(13)] +=  -7.9366827390519274e+00 * (j_temp * 9.2803892982949643e-01 + 1.9280389298294964e+00 * pres_mod_temp);
  jac[INDEX(15)] =  8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01 + 1.9280389298294964e+00 * pres_mod_temp);
  jac[INDEX(22)] =  -(j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(23)] +=  -1.5873365478103855e+01 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(25)] =  1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(32)] =  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * conc[INDEX(1)]));
  jac[INDEX(33)] +=  -(j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * conc[INDEX(1)]));
  jac[INDEX(35)] =  1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * conc[INDEX(1)]));
  jac[INDEX(42)] =  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(43)] +=  -5.0000000000000000e-01 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(45)] =  5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(52)] =  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (-kr));
  jac[INDEX(53)] +=  -9.4073500029987056e-01 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (-kr));
  jac[INDEX(55)] =  (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (-kr));
  jac[INDEX(62)] =  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01 + 5.3569067463235553e+00 * pres_mod_temp);
  jac[INDEX(63)] +=  -8.8810165592763468e-01 * (j_temp * 3.5690674632355568e-01 + 5.3569067463235553e+00 * pres_mod_temp);
  jac[INDEX(65)] =  9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01 + 5.3569067463235553e+00 * pres_mod_temp);
  jac[INDEX(72)] =  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(73)] +=  -4.8473130033441653e-01 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(75)] =  5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(82)] =  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(83)] +=  -4.7036750014993528e-01 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(85)] =  5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(92)] =  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01 + -7.2602775520927776e-01 * pres_mod_temp);
  jac[INDEX(93)] +=  -4.0050565735456090e-01 * (j_temp * -4.2602775520927771e-01 + -7.2602775520927776e-01 * pres_mod_temp);
  jac[INDEX(95)] =  4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01 + -7.2602775520927776e-01 * pres_mod_temp);

  shared_temp[threadIdx.x + 1 * blockDim.x] = fwd_rates[INDEX(2)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(2)];
  //partial of rxn 2 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 1 * blockDim.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * (2.7000000000000002e+00 + (3.1501544760183588e+03 / T)) + shared_temp[threadIdx.x + 1 * blockDim.x] * -1.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(1)] + dBdT[INDEX(4)] - dBdT[INDEX(0)] - dBdT[INDEX(2)])))) * rho_inv;
  jac[INDEX(1)] = -j_temp * 2.0158800000000001e+00;
  jac[INDEX(2)] += j_temp * 1.0079400000000001e+00;
  jac[INDEX(3)] += -j_temp * 1.5999400000000000e+01;
  jac[INDEX(5)] += j_temp * 1.7007339999999999e+01;

  //partial of rxn 2 wrt species
  if (T <= 1000.0) {
    Kc = (-4.2649692190000001e+00 + 9.7941720999999982e-01 * logT + T * (-3.5512597146473328e-03 + T * (2.9088375746673469e-06 + T * (-1.4925230406415989e-09 + 3.3137862995361342e-13 * T))) - 8.8441643299999987e+02 / T);
  } else {
    Kc = (2.7645101560000001e+00 + -3.1381229999999993e-01 * logT + T * (3.4190313985785133e-04 + T * (-6.9149998790634198e-08 + T * (8.4698344637373041e-12 + -4.7563182059013212e-16 * T))) - 1.0648967219999986e+03 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(3.6558396000357360e+00 + 2.7 * logT - (3.1501544760183588e+03 / T));
  kr = kf / Kc;
  jac[INDEX(11)] =  -(j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(12)] +=  5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(13)] +=  -7.9366827390519274e+00 * (j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(15)] +=  8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(21)] =  -2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(22)] +=  (j_temp * 9.6401946491474821e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(23)] +=  -1.5873365478103855e+01 * (j_temp * 9.6401946491474821e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(25)] +=  1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(31)] =  -1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(0)]);
  jac[INDEX(32)] +=  6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(0)]);
  jac[INDEX(33)] +=  -(j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(0)]);
  jac[INDEX(35)] +=  1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(0)]);
  jac[INDEX(41)] =  -6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(42)] +=  3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(43)] +=  -5.0000000000000000e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(45)] +=  5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(51)] =  -1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(1)]);
  jac[INDEX(52)] +=  5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(1)]);
  jac[INDEX(53)] +=  -9.4073500029987056e-01 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(1)]);
  jac[INDEX(55)] +=  (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(1)]);
  jac[INDEX(61)] =  -1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(62)] +=  5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(63)] +=  -8.8810165592763468e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(65)] +=  9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(71)] =  -6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(72)] +=  3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(73)] +=  -4.8473130033441653e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(75)] +=  5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(81)] =  -5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(82)] +=  2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(83)] +=  -4.7036750014993528e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(85)] +=  5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(91)] =  -5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(92)] +=  2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(93)] +=  -4.0050565735456090e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x] = fwd_rates[INDEX(3)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(3)];
  //partial of rxn 3 wrt T
  j_temp = ((1.0 / T) * (shared_temp[threadIdx.x] * -1.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(3)] + dBdT[INDEX(4)] - dBdT[INDEX(6)] - dBdT[INDEX(2)])))) * rho_inv;
  jac[INDEX(3)] += -j_temp * 1.5999400000000000e+01;
  jac[INDEX(4)] += j_temp * 3.1998799999999999e+01;
  jac[INDEX(5)] += j_temp * 1.7007339999999999e+01;
  jac[INDEX(7)] = -j_temp * 3.3006740000000001e+01;

  //partial of rxn 3 wrt species
  if (T <= 1000.0) {
    Kc = (-2.5192523180000004e+00 + 3.0440668000000004e-01 * logT + T * (1.3151938349999998e-03 + T * (-2.2226854399999996e-06 + T * (1.4035022683333334e-09 + -3.3985339400000004e-13 * T))) - -2.6865930240000002e+04 / T);
  } else {
    Kc = (1.5716927700000003e+00 + -2.1120617000000053e-01 * logT + T * (-6.1164380150000074e-05 + T * (-6.6252916499999993e-09 + T * (1.4413174416666625e-12 + -1.8401260499999993e-17 * T))) - -2.6559236532999999e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = 20000000000.000004;
  kr = kf / Kc;
  jac[INDEX(13)] +=  -7.9366827390519274e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(14)] +=  1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(15)] +=  8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] =  -1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(23)] +=  -1.5873365478103855e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(24)] +=  3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(25)] +=  1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(27)] =  -3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(33)] +=  -(j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(34)] +=  2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(35)] +=  1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(37)] =  -2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(43)] +=  -5.0000000000000000e-01 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(44)] +=  (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(45)] +=  5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(47)] =  -1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(53)] +=  -9.4073500029987056e-01 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(54)] +=  1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(55)] +=  (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(57)] =  -1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(63)] +=  -8.8810165592763468e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(64)] +=  1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(65)] +=  9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] =  -1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(73)] +=  -4.8473130033441653e-01 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(74)] +=  9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(75)] +=  5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(77)] =  -(j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(83)] +=  -4.7036750014993528e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(84)] +=  9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(85)] +=  5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(87)] =  -9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(93)] +=  -4.0050565735456090e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(94)] +=  8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] =  -8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 1 * blockDim.x] = fwd_rates[INDEX(4)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(4)];
  //partial of rxn 4 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 1 * blockDim.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * (2.0000000000000000e+00 + (2.0128782594366510e+03 / T)) + shared_temp[threadIdx.x + 1 * blockDim.x] * -1.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(6)] + dBdT[INDEX(4)] - dBdT[INDEX(7)] - dBdT[INDEX(2)])))) * rho_inv;
  jac[INDEX(3)] += -j_temp * 1.5999400000000000e+01;
  jac[INDEX(5)] += j_temp * 1.7007339999999999e+01;
  jac[INDEX(7)] += j_temp * 3.3006740000000001e+01;
  jac[INDEX(8)] = -j_temp * 3.4014679999999998e+01;

  //partial of rxn 4 wrt species
  if (T <= 1000.0) {
    Kc = (-2.7236808580000003e+00 + 8.4943364999999993e-01 * logT + T * (-1.6641483864999998e-03 + T * (3.9993224166666590e-07 + T * (-3.7697932500000234e-11 + -4.0418699999999293e-15 * T))) - -7.5097885000000024e+03 / T);
  } else {
    Kc = (1.8562804999999960e-01 + 3.7567494000000057e-01 * logT + T * (-1.0170464901500000e-03 + T * (2.2538181151666665e-07 + T * (-2.7905665974999994e-11 + 1.4256438845000000e-15 * T))) - -7.3852776869999980e+03 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(9.1726385047921717e+00 + 2.0 * logT - (2.0128782594366510e+03 / T));
  kr = kf / Kc;
  jac[INDEX(13)] +=  -7.9366827390519274e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(15)] +=  8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(18)] =  -1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(23)] +=  -1.5873365478103855e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(25)] +=  1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(27)] +=  3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(28)] =  -3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(33)] +=  -(j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(35)] +=  1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(37)] +=  2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(38)] =  -2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(43)] +=  -5.0000000000000000e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(45)] +=  5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(47)] +=  1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(48)] =  -1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(53)] +=  -9.4073500029987056e-01 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(55)] +=  (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(57)] +=  1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(58)] =  -2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(63)] +=  -8.8810165592763468e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(65)] +=  9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] +=  1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(68)] =  -1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(73)] +=  -4.8473130033441653e-01 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(75)] +=  5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(77)] +=  (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(78)] =  -1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(83)] +=  -4.7036750014993528e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(85)] +=  5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(87)] +=  9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(88)] =  -(j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(93)] +=  -4.0050565735456090e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(98)] =  -8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x] = pres_mod[INDEX(2)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = fwd_rates[INDEX(5)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(5)];
  //partial of rxn 5 wrt T
  j_temp = ((-shared_temp[threadIdx.x] * (shared_temp[threadIdx.x + 2 * blockDim.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) / T) + (shared_temp[threadIdx.x] / T) * ((shared_temp[threadIdx.x + 2 * blockDim.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * (-8.5999999999999999e-01) + shared_temp[threadIdx.x + 2 * blockDim.x] * -1.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-T * (dBdT[INDEX(6)] - dBdT[INDEX(1)] - dBdT[INDEX(3)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(4)] += -j_temp * 3.1998799999999999e+01;
  jac[INDEX(7)] += j_temp * 3.3006740000000001e+01;

  //partial of rxn 5 wrt species
  if (T <= 1000.0) {
    Kc = (2.4863279230000002e+00 + -9.8065834999999968e-01 * logT + T * (-8.7619317535266635e-04 + T * (1.8851645153326530e-06 + T * (-1.2162578593584016e-09 + 3.0242614354638670e-13 * T))) - -2.4114908299999999e+04 / T);
  } else {
    Kc = (5.4388072400000009e-01 + -7.6532694999999951e-01 * logT + T * (3.7836630654214864e-04 + T * (2.0718083807300866e-08 + T * (-7.9353483554039681e-12 + 5.4404627009013201e-16 * T))) - -2.4273345466999999e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  pres_mod_temp = (shared_temp[threadIdx.x + 2 * blockDim.x] - shared_temp[threadIdx.x + 3 * blockDim.x]);
  j_temp = -mw_avg * rho_inv * shared_temp[threadIdx.x] * (3.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(2.8660640533109706e+01 - 0.86 * logT);
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01 + pres_mod_temp);
  jac[INDEX(14)] +=  -1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01 + pres_mod_temp);
  jac[INDEX(17)] +=  1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01 + pres_mod_temp);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + pres_mod_temp + shared_temp[threadIdx.x] * (kf * conc[INDEX(3)]));
  jac[INDEX(24)] +=  -3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01 + pres_mod_temp + shared_temp[threadIdx.x] * (kf * conc[INDEX(3)]));
  jac[INDEX(27)] +=  3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + pres_mod_temp + shared_temp[threadIdx.x] * (kf * conc[INDEX(3)]));
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01 + pres_mod_temp);
  jac[INDEX(34)] +=  -2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01 + pres_mod_temp);
  jac[INDEX(37)] +=  2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01 + pres_mod_temp);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01 + shared_temp[threadIdx.x] * (kf * conc[INDEX(1)]));
  jac[INDEX(44)] +=  -(j_temp * -1.4226436701188128e-01 + shared_temp[threadIdx.x] * (kf * conc[INDEX(1)]));
  jac[INDEX(47)] +=  1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01 + shared_temp[threadIdx.x] * (kf * conc[INDEX(1)]));
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01 + pres_mod_temp);
  jac[INDEX(54)] +=  -1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01 + pres_mod_temp);
  jac[INDEX(57)] +=  1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01 + pres_mod_temp);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(64)] +=  -1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] +=  1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 + pres_mod_temp + shared_temp[threadIdx.x] * (-kr));
  jac[INDEX(74)] +=  -9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 + pres_mod_temp + shared_temp[threadIdx.x] * (-kr));
  jac[INDEX(77)] +=  (j_temp * -1.7824490209713328e-01 + pres_mod_temp + shared_temp[threadIdx.x] * (-kr));
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01 + pres_mod_temp);
  jac[INDEX(84)] +=  -9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01 + pres_mod_temp);
  jac[INDEX(87)] +=  9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01 + pres_mod_temp);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(94)] +=  -8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 1 * blockDim.x] = conc[INDEX(3)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = fwd_rates[INDEX(6)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(6)];
  //partial of rxn 6 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 2 * blockDim.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * (-1.2400000000000000e+00) + shared_temp[threadIdx.x + 2 * blockDim.x] * -2.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(6)] - dBdT[INDEX(3)] - dBdT[INDEX(1)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(4)] += -j_temp * 3.1998799999999999e+01;
  jac[INDEX(7)] += j_temp * 3.3006740000000001e+01;

  //partial of rxn 6 wrt species
  if (T <= 1000.0) {
    Kc = (2.4863279230000002e+00 + -9.8065834999999968e-01 * logT + T * (-8.7619317535266635e-04 + T * (1.8851645153326530e-06 + T * (-1.2162578593584016e-09 + 3.0242614354638670e-13 * T))) - -2.4114908299999999e+04 / T);
  } else {
    Kc = (5.4388072400000009e-01 + -7.6532694999999951e-01 * logT + T * (3.7836630654214864e-04 + T * (2.0718083807300866e-08 + T * (-7.9353483554039681e-12 + 5.4404627009013201e-16 * T))) - -2.4273345466999999e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  j_temp = -mw_avg * rho_inv * (3.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(3.0665974102635822e+01 - 1.24 * logT);
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(14)] +=  -1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(24)] +=  -3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(27)] +=  3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(34)] +=  -2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01 + kf * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x] * conc[INDEX(1)] - kr * conc[INDEX(6)]);
  jac[INDEX(44)] +=  -(j_temp * -1.4226436701188128e-01 + kf * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x] * conc[INDEX(1)] - kr * conc[INDEX(6)]);
  jac[INDEX(47)] +=  1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01 + kf * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x] * conc[INDEX(1)] - kr * conc[INDEX(6)]);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(54)] +=  -1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(57)] +=  1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(64)] +=  -1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] +=  1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 - kr * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(74)] +=  -9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 - kr * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(77)] +=  (j_temp * -1.7824490209713328e-01 - kr * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(84)] +=  -9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(87)] +=  9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(94)] +=  -8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x] = fwd_rates[INDEX(7)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = rev_rates[INDEX(7)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = conc[INDEX(5)];
  //partial of rxn 7 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x] - shared_temp[threadIdx.x + 2 * blockDim.x]) * (-7.6000000000000001e-01) + shared_temp[threadIdx.x] * -2.0 - shared_temp[threadIdx.x + 2 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(6)] - dBdT[INDEX(1)] - dBdT[INDEX(3)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(4)] += -j_temp * 3.1998799999999999e+01;
  jac[INDEX(7)] += j_temp * 3.3006740000000001e+01;

  //partial of rxn 7 wrt species
  if (T <= 1000.0) {
    Kc = (2.4863279230000002e+00 + -9.8065834999999968e-01 * logT + T * (-8.7619317535266635e-04 + T * (1.8851645153326530e-06 + T * (-1.2162578593584016e-09 + 3.0242614354638670e-13 * T))) - -2.4114908299999999e+04 / T);
  } else {
    Kc = (5.4388072400000009e-01 + -7.6532694999999951e-01 * logT + T * (3.7836630654214864e-04 + T * (2.0718083807300866e-08 + T * (-7.9353483554039681e-12 + 5.4404627009013201e-16 * T))) - -2.4273345466999999e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  j_temp = -mw_avg * rho_inv * (3.0 * shared_temp[threadIdx.x] - 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]);
  kf = exp(3.0052277738640093e+01 - 0.76 * logT);
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(14)] +=  -1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x]);
  jac[INDEX(24)] +=  -3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x]);
  jac[INDEX(27)] +=  3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x]);
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(34)] +=  -2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 3 * blockDim.x]);
  jac[INDEX(44)] +=  -(j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 3 * blockDim.x]);
  jac[INDEX(47)] +=  1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 3 * blockDim.x]);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(54)] +=  -1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(57)] +=  1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01 + kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]);
  jac[INDEX(64)] +=  -1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01 + kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]);
  jac[INDEX(67)] +=  1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01 + kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 - kr * shared_temp[threadIdx.x + 3 * blockDim.x]);
  jac[INDEX(74)] +=  -9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 - kr * shared_temp[threadIdx.x + 3 * blockDim.x]);
  jac[INDEX(77)] +=  (j_temp * -1.7824490209713328e-01 - kr * shared_temp[threadIdx.x + 3 * blockDim.x]);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(84)] +=  -9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(87)] +=  9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(94)] +=  -8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 2 * blockDim.x] = fwd_rates[INDEX(8)];
  //partial of rxn 8 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 2 * blockDim.x] - rev_rates[INDEX(8)]) * (-1.2400000000000000e+00) + shared_temp[threadIdx.x + 2 * blockDim.x] * -2.0 - rev_rates[INDEX(8)] * (-1.0 + -T * (dBdT[INDEX(6)] - dBdT[INDEX(1)] - dBdT[INDEX(3)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(4)] += -j_temp * 3.1998799999999999e+01;
  jac[INDEX(7)] += j_temp * 3.3006740000000001e+01;

  //partial of rxn 8 wrt species
  if (T <= 1000.0) {
    Kc = (2.4863279230000002e+00 + -9.8065834999999968e-01 * logT + T * (-8.7619317535266635e-04 + T * (1.8851645153326530e-06 + T * (-1.2162578593584016e-09 + 3.0242614354638670e-13 * T))) - -2.4114908299999999e+04 / T);
  } else {
    Kc = (5.4388072400000009e-01 + -7.6532694999999951e-01 * logT + T * (3.7836630654214864e-04 + T * (2.0718083807300866e-08 + T * (-7.9353483554039681e-12 + 5.4404627009013201e-16 * T))) - -2.4273345466999999e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  j_temp = -mw_avg * rho_inv * (3.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 2.0 * rev_rates[INDEX(8)]);
  kf = exp(3.0889117653950031e+01 - 1.24 * logT);
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01 - 7.1961070170503627e-02 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(14)] +=  -1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01 - 7.1961070170503627e-02 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(17)] +=  1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01 - 7.1961070170503627e-02 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * conc[INDEX(9)] - 3.5980535085251814e-02 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(24)] +=  -3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * conc[INDEX(9)] - 3.5980535085251814e-02 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(27)] +=  3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * conc[INDEX(9)] - 3.5980535085251814e-02 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01 - 5.7113218350594064e-01 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(34)] +=  -2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01 - 5.7113218350594064e-01 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(37)] +=  2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01 - 5.7113218350594064e-01 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)] * conc[INDEX(9)] - 1.1422643670118813e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(44)] +=  -(j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)] * conc[INDEX(9)] - 1.1422643670118813e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(47)] +=  1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)] * conc[INDEX(9)] - 1.1422643670118813e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01 - 6.0711271859119242e-01 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(54)] +=  -1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01 - 6.0711271859119242e-01 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(57)] +=  1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01 - 6.0711271859119242e-01 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01 - 6.4309325367644432e-01 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(64)] +=  -1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01 - 6.4309325367644432e-01 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(67)] +=  1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01 - 6.4309325367644432e-01 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(9)] - 1.1782449020971333e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(74)] +=  -9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(9)] - 1.1782449020971333e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(77)] +=  (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(9)] - 1.1782449020971333e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01 - 1.2142254371823848e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(84)] +=  -9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01 - 1.2142254371823848e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(87)] +=  9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01 - 1.2142254371823848e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01 - 1.4260277552092777e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(94)] +=  -8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01 - 1.4260277552092777e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));
  jac[INDEX(97)] +=  8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01 - 1.4260277552092777e+00 * (kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]));

  shared_temp[threadIdx.x] = fwd_rates[INDEX(9)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(9)];
  //partial of rxn 9 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * (-8.0000000000000004e-01) + shared_temp[threadIdx.x] * -2.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(6)] - dBdT[INDEX(1)] - dBdT[INDEX(3)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(4)] += -j_temp * 3.1998799999999999e+01;
  jac[INDEX(7)] += j_temp * 3.3006740000000001e+01;

  //partial of rxn 9 wrt species
  if (T <= 1000.0) {
    Kc = (2.4863279230000002e+00 + -9.8065834999999968e-01 * logT + T * (-8.7619317535266635e-04 + T * (1.8851645153326530e-06 + T * (-1.2162578593584016e-09 + 3.0242614354638670e-13 * T))) - -2.4114908299999999e+04 / T);
  } else {
    Kc = (5.4388072400000009e-01 + -7.6532694999999951e-01 * logT + T * (3.7836630654214864e-04 + T * (2.0718083807300866e-08 + T * (-7.9353483554039681e-12 + 5.4404627009013201e-16 * T))) - -2.4273345466999999e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  j_temp = -mw_avg * rho_inv * (3.0 * shared_temp[threadIdx.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(2.7274346171989816e+01 - 0.8 * logT);
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(14)] +=  -1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * conc[INDEX(8)]);
  jac[INDEX(24)] +=  -3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * conc[INDEX(8)]);
  jac[INDEX(27)] +=  3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x] * conc[INDEX(8)]);
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(34)] +=  -2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)] * conc[INDEX(8)]);
  jac[INDEX(44)] +=  -(j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)] * conc[INDEX(8)]);
  jac[INDEX(47)] +=  1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)] * conc[INDEX(8)]);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(54)] +=  -1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(57)] +=  1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(64)] +=  -1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] +=  1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(8)]);
  jac[INDEX(74)] +=  -9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(8)]);
  jac[INDEX(77)] +=  (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(8)]);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(84)] +=  -9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(87)] +=  9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01 + kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]);
  jac[INDEX(94)] +=  -8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01 + kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]);
  jac[INDEX(97)] +=  8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01 + kf * conc[INDEX(1)] * shared_temp[threadIdx.x + 1 * blockDim.x] - kr * conc[INDEX(6)]);

  shared_temp[threadIdx.x + 2 * blockDim.x] = fwd_rates[INDEX(10)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(10)];
  //partial of rxn 10 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 2 * blockDim.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * (-6.7069999999999996e-01 + (8.5753646047649927e+03 / T)) + shared_temp[threadIdx.x + 2 * blockDim.x] * -1.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(2)] + dBdT[INDEX(4)] - dBdT[INDEX(1)] - dBdT[INDEX(3)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(3)] += j_temp * 1.5999400000000000e+01;
  jac[INDEX(4)] += -j_temp * 3.1998799999999999e+01;
  jac[INDEX(5)] += j_temp * 1.7007339999999999e+01;

  //partial of rxn 10 wrt species
  if (T <= 1000.0) {
    Kc = (-2.1408110450000004e+00 + 8.7782617000000007e-01 * logT + T * (-1.3419511003526666e-03 + T * (2.3561672699932007e-07 + T * (-2.7325373525067973e-11 + 1.1652302046386607e-14 * T))) - 8.3276234200000017e+03 / T);
  } else {
    Kc = (4.3747157639999994e+00 + -1.2022939999999949e-01 * logT + T * (-5.1031595730785127e-04 + T * (1.5440338995730085e-07 + T * (-2.5619540480403969e-11 + 1.7320676705901320e-15 * T))) - 8.6910339200000017e+03 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(3.0908165848920724e+01 - 0.6707 * logT - (8.5753646047649927e+03 / T));
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(13)] +=  7.9366827390519274e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(14)] +=  -1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(15)] +=  8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(23)] +=  1.5873365478103855e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(24)] +=  -3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(25)] +=  1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(33)] +=  (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(34)] +=  -2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(35)] +=  1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(4)]);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)]);
  jac[INDEX(43)] +=  5.0000000000000000e-01 * (j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)]);
  jac[INDEX(44)] +=  -(j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)]);
  jac[INDEX(45)] +=  5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01 + kf * conc[INDEX(1)]);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(53)] +=  9.4073500029987056e-01 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(54)] +=  -1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(55)] +=  (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(63)] +=  8.8810165592763468e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(64)] +=  -1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(65)] +=  9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(73)] +=  4.8473130033441653e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(74)] +=  -9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(75)] +=  5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(83)] +=  4.7036750014993528e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(84)] +=  -9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(85)] +=  5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(93)] +=  4.0050565735456090e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(94)] +=  -8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x] = pres_mod[INDEX(3)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = conc[INDEX(1)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(11)];
  //partial of rxn 11 wrt T
  j_temp = ((-shared_temp[threadIdx.x] * (shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(11)]) / T) + (shared_temp[threadIdx.x] / T) * ((shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(11)]) * (-1.0000000000000000e+00) + shared_temp[threadIdx.x + 3 * blockDim.x] * -1.0 - rev_rates[INDEX(11)] * (-T * (dBdT[INDEX(0)] - 2.0 * dBdT[INDEX(1)])))) * rho_inv;
  jac[INDEX(1)] += j_temp * 2.0158800000000001e+00;
  jac[INDEX(2)] += j_temp * -2.0 * 1.0079400000000001e+00;

  //partial of rxn 11 wrt species
  if (T <= 1000.0) {
    Kc = (4.2320448239999999e+00 + -1.6556688799999999e+00 * logT + T * (3.9902603742946671e-03 + T * (-3.2463584993346931e-06 + T * (1.6797674496165307e-09 + -3.6880588040722682e-13 * T))) - -5.1865254972999996e+04 / T);
  } else {
    Kc = (-6.4893666199999966e-01 + -6.6272082000000010e-01 * logT + T * (-2.4701213465702698e-05 + T * (8.3242790947935064e-08 + T * (-1.4963865377474610e-11 + 1.0012768301802643e-15 * T))) - -5.1897478722000000e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  pres_mod_temp = (shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(11)]);
  j_temp = -mw_avg * rho_inv * shared_temp[threadIdx.x] * (3.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 2.0 * rev_rates[INDEX(11)]);
  kf = exp(2.7631021115928547e+01 - 1.0 * logT);
  kr = kf / Kc;
  jac[INDEX(11)] +=  (j_temp * 9.2803892982949643e-01 + -7.1961070170503627e-02 * pres_mod_temp + shared_temp[threadIdx.x] * (-kr));
  jac[INDEX(12)] +=  -(j_temp * 9.2803892982949643e-01 + -7.1961070170503627e-02 * pres_mod_temp + shared_temp[threadIdx.x] * (-kr));
  jac[INDEX(21)] +=  2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp + shared_temp[threadIdx.x] * (kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(22)] +=  -2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp + shared_temp[threadIdx.x] * (kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(31)] +=  1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp);
  jac[INDEX(32)] +=  -1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp);
  jac[INDEX(41)] +=  6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(42)] +=  -6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(51)] +=  1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp);
  jac[INDEX(52)] +=  -1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp);
  jac[INDEX(61)] +=  1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01 + -6.4309325367644432e-01 * pres_mod_temp);
  jac[INDEX(62)] +=  -1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01 + -6.4309325367644432e-01 * pres_mod_temp);
  jac[INDEX(71)] +=  6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(72)] +=  -6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(81)] +=  5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(82)] +=  -5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(91)] +=  5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01 + -7.9602775520927771e-01 * pres_mod_temp);
  jac[INDEX(92)] +=  -5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01 + -7.9602775520927771e-01 * pres_mod_temp);

  shared_temp[threadIdx.x + 1 * blockDim.x] = conc[INDEX(0)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(12)];
  //partial of rxn 12 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(12)]) * (-5.9999999999999998e-01) + shared_temp[threadIdx.x + 3 * blockDim.x] * -2.0 - rev_rates[INDEX(12)] * (-1.0 + -T * (dBdT[INDEX(0)] - 2.0 * dBdT[INDEX(1)])))) * rho_inv;
  jac[INDEX(1)] += j_temp * 2.0158800000000001e+00;
  jac[INDEX(2)] += j_temp * -2.0 * 1.0079400000000001e+00;

  //partial of rxn 12 wrt species
  if (T <= 1000.0) {
    Kc = (4.2320448239999999e+00 + -1.6556688799999999e+00 * logT + T * (3.9902603742946671e-03 + T * (-3.2463584993346931e-06 + T * (1.6797674496165307e-09 + -3.6880588040722682e-13 * T))) - -5.1865254972999996e+04 / T);
  } else {
    Kc = (-6.4893666199999966e-01 + -6.6272082000000010e-01 * logT + T * (-2.4701213465702698e-05 + T * (8.3242790947935064e-08 + T * (-1.4963865377474610e-11 + 1.0012768301802643e-15 * T))) - -5.1897478722000000e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  j_temp = -mw_avg * rho_inv * (3.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 2.0 * rev_rates[INDEX(12)]);
  kf = exp(2.5223075507276675e+01 - 0.6 * logT);
  kr = kf / Kc;
  jac[INDEX(11)] +=  (j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] - kr * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(12)] +=  -(j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] - kr * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(21)] +=  2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(22)] +=  -2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(31)] +=  1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(32)] +=  -1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(41)] +=  6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(42)] +=  -6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(51)] +=  1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(52)] +=  -1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(61)] +=  1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(62)] +=  -1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(71)] +=  6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(72)] +=  -6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(81)] +=  5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(82)] +=  -5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(91)] +=  5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(92)] +=  -5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x] = fwd_rates[INDEX(13)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(13)];
  //partial of rxn 13 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * (-1.2500000000000000e+00) + shared_temp[threadIdx.x] * -2.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(0)] - 2.0 * dBdT[INDEX(1)])))) * rho_inv;
  jac[INDEX(1)] += j_temp * 2.0158800000000001e+00;
  jac[INDEX(2)] += j_temp * -2.0 * 1.0079400000000001e+00;

  //partial of rxn 13 wrt species
  if (T <= 1000.0) {
    Kc = (4.2320448239999999e+00 + -1.6556688799999999e+00 * logT + T * (3.9902603742946671e-03 + T * (-3.2463584993346931e-06 + T * (1.6797674496165307e-09 + -3.6880588040722682e-13 * T))) - -5.1865254972999996e+04 / T);
  } else {
    Kc = (-6.4893666199999966e-01 + -6.6272082000000010e-01 * logT + T * (-2.4701213465702698e-05 + T * (8.3242790947935064e-08 + T * (-1.4963865377474610e-11 + 1.0012768301802643e-15 * T))) - -5.1897478722000000e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  j_temp = -mw_avg * rho_inv * (3.0 * shared_temp[threadIdx.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(3.1725365678150649e+01 - 1.25 * logT);
  kr = kf / Kc;
  jac[INDEX(11)] +=  (j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(12)] +=  -(j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(21)] +=  2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x] * conc[INDEX(5)]);
  jac[INDEX(22)] +=  -2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x] * conc[INDEX(5)]);
  jac[INDEX(31)] +=  1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(32)] +=  -1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(41)] +=  6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(42)] +=  -6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(51)] +=  1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(52)] +=  -1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(61)] +=  1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] - kr * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(62)] +=  -1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] - kr * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(71)] +=  6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(72)] +=  -6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(81)] +=  5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(82)] +=  -5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(91)] +=  5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(92)] +=  -5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 3 * blockDim.x] = pres_mod[INDEX(4)];
  //partial of rxn 14 wrt T
  j_temp = ((-shared_temp[threadIdx.x + 3 * blockDim.x] * (fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]) / T) + (shared_temp[threadIdx.x + 3 * blockDim.x] / T) * ((fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]) * (-2.0000000000000000e+00) + fwd_rates[INDEX(14)] * -1.0 - rev_rates[INDEX(14)] * (-T * (dBdT[INDEX(5)] - dBdT[INDEX(1)] - dBdT[INDEX(4)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(5)] += -j_temp * 1.7007339999999999e+01;
  jac[INDEX(6)] = j_temp * 1.8015280000000001e+01;

  //partial of rxn 14 wrt species
  if (T <= 1000.0) {
    Kc = (1.9949509729999999e+00 + -1.2933748700000001e+00 * logT + T * (1.8244170964733350e-04 + T * (3.1707728366598661e-07 + T * (-1.3390310769173469e-10 + 2.0393173546386613e-14 * T))) - -5.9382467160000000e+04 / T);
  } else {
    Kc = (3.4956521039999999e+00 + -1.5588951899999999e+00 * logT + T * (8.1424417354214874e-04 + T * (-4.8429627026032460e-08 + T * (-7.5798555540397048e-13 + 2.5394305509013217e-16 * T))) - -5.9336614000000001e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  pres_mod_temp = (fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]);
  j_temp = -mw_avg * rho_inv * shared_temp[threadIdx.x + 3 * blockDim.x] * (3.0 * fwd_rates[INDEX(14)] - 2.0 * rev_rates[INDEX(14)]);
  kf = exp(3.7629818848269004e+01 - 2.0 * logT);
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01 + 6.5803892982949641e-01 * pres_mod_temp);
  jac[INDEX(15)] +=  -8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01 + 6.5803892982949641e-01 * pres_mod_temp);
  jac[INDEX(16)] =  8.9366827390519283e+00 * (j_temp * 9.2803892982949643e-01 + 6.5803892982949641e-01 * pres_mod_temp);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * conc[INDEX(4)]));
  jac[INDEX(25)] +=  -1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * conc[INDEX(4)]));
  jac[INDEX(26)] =  1.7873365478103857e+01 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * conc[INDEX(4)]));
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp);
  jac[INDEX(35)] +=  -1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp);
  jac[INDEX(36)] =  1.1259972248959336e+00 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(45)] +=  -5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(46)] =  5.6299861244796678e-01 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(55)] +=  -(j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(56)] =  1.0592649997001296e+00 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (kf * shared_temp[threadIdx.x + 2 * blockDim.x]));
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01 + 3.0069067463235557e+00 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (-kr));
  jac[INDEX(65)] +=  -9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01 + 3.0069067463235557e+00 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (-kr));
  jac[INDEX(66)] =  (j_temp * 3.5690674632355568e-01 + 3.0069067463235557e+00 * pres_mod_temp + shared_temp[threadIdx.x + 3 * blockDim.x] * (-kr));
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(75)] +=  -5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(76)] =  5.4580609899675037e-01 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(85)] +=  -5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(86)] =  5.2963249985006478e-01 * (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01 + -1.0460277552092778e+00 * pres_mod_temp);
  jac[INDEX(95)] +=  -4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01 + -1.0460277552092778e+00 * pres_mod_temp);
  jac[INDEX(96)] =  4.5096825873635726e-01 * (j_temp * -4.2602775520927771e-01 + -1.0460277552092778e+00 * pres_mod_temp);

  shared_temp[threadIdx.x + 1 * blockDim.x] = fwd_rates[INDEX(15)];
  shared_temp[threadIdx.x] = rev_rates[INDEX(15)];
  //partial of rxn 15 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 1 * blockDim.x] - shared_temp[threadIdx.x]) * ((3.3766032802049818e+02 / T)) + shared_temp[threadIdx.x + 1 * blockDim.x] * -1.0 - shared_temp[threadIdx.x] * (-1.0 + -T * (dBdT[INDEX(5)] + dBdT[INDEX(2)] - dBdT[INDEX(1)] - dBdT[INDEX(6)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(3)] += j_temp * 1.5999400000000000e+01;
  jac[INDEX(6)] += j_temp * 1.8015280000000001e+01;
  jac[INDEX(7)] += -j_temp * 3.3006740000000001e+01;

  //partial of rxn 15 wrt species
  if (T <= 1000.0) {
    Kc = (-2.6321879950000002e+00 + 5.6510965000000057e-01 * logT + T * (-2.8331621535266673e-04 + T * (-1.3324705046673464e-06 + T * (1.0550293781415989e-09 + -2.7038066795361348e-13 * T))) - -2.6939935439999997e+04 / T);
  } else {
    Kc = (7.3264871439999988e+00 + -9.1379763999999986e-01 * logT + T * (-7.4438090307851279e-05 + T * (8.5255679123967526e-08 + T * (-1.8442177680403972e-11 + 1.4419644555901321e-15 * T))) - -2.6372234613000001e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 2.0 * shared_temp[threadIdx.x]);
  kf = exp(2.2102031931645509e+01 - (3.3766032802049818e+02 / T));
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(13)] +=  7.9366827390519274e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(16)] +=  8.9366827390519283e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  -1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(23)] +=  1.5873365478103855e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(26)] +=  1.7873365478103857e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(27)] +=  -3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(33)] +=  (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(36)] +=  1.1259972248959336e+00 * (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(37)] +=  -2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(43)] +=  5.0000000000000000e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(46)] +=  5.6299861244796678e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(47)] +=  -1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(53)] +=  9.4073500029987056e-01 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(56)] +=  1.0592649997001296e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(57)] +=  -1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(63)] +=  8.8810165592763468e-01 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(66)] +=  (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(67)] +=  -1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(73)] +=  4.8473130033441653e-01 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(76)] +=  5.4580609899675037e-01 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(77)] +=  -(j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(83)] +=  4.7036750014993528e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(86)] +=  5.2963249985006478e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(87)] +=  -9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(93)] +=  4.0050565735456090e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(96)] +=  4.5096825873635726e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  -8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(16)];
  //partial of rxn 16 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(16)]) * ((5.3743849526958581e+02 / T)) + shared_temp[threadIdx.x + 3 * blockDim.x] * -1.0 - rev_rates[INDEX(16)] * (-1.0 + -T * (dBdT[INDEX(0)] + dBdT[INDEX(3)] - dBdT[INDEX(1)] - dBdT[INDEX(6)])))) * rho_inv;
  jac[INDEX(1)] += j_temp * 2.0158800000000001e+00;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(4)] += j_temp * 3.1998799999999999e+01;
  jac[INDEX(7)] += -j_temp * 3.3006740000000001e+01;

  //partial of rxn 16 wrt species
  if (T <= 1000.0) {
    Kc = (1.7457169009999998e+00 + -6.7501052999999978e-01 * logT + T * (4.8664535496473326e-03 + T * (-5.1315230146673465e-06 + T * (2.8960253089749321e-09 + -6.7123202395361341e-13 * T))) - -2.7750346673000000e+04 / T);
  } else {
    Kc = (-1.1928173859999998e+00 + 1.0260612999999941e-01 * logT + T * (-4.0306752000785135e-04 + T * (6.2524707140634199e-08 + T * (-7.0285170220706416e-12 + 4.5723056009013213e-16 * T))) - -2.7624133254999997e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 2.0 * rev_rates[INDEX(16)]);
  kf = exp(2.4525473976367351e+01 - (5.3743849526958581e+02 / T));
  kr = kf / Kc;
  jac[INDEX(11)] +=  (j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(14)] +=  1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(17)] +=  -1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(21)] +=  2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(24)] +=  3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(27)] +=  -3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(31)] +=  1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(34)] +=  2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  -2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(41)] +=  6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(0)]);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(0)]);
  jac[INDEX(44)] +=  (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(0)]);
  jac[INDEX(47)] +=  -1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(0)]);
  jac[INDEX(51)] +=  1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(54)] +=  1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(57)] +=  -1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(61)] +=  1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(64)] +=  1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] +=  -1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(71)] +=  6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(74)] +=  9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(77)] +=  -(j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(81)] +=  5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(84)] +=  9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(87)] +=  -9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(91)] +=  5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(94)] +=  8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  -8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 1 * blockDim.x] = conc[INDEX(4)];
  shared_temp[threadIdx.x] = fwd_rates[INDEX(17)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(17)];
  //partial of rxn 17 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * ((3.1954442368556835e+02 / T)) + shared_temp[threadIdx.x] * -1.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (2.0 * dBdT[INDEX(4)] - dBdT[INDEX(1)] - dBdT[INDEX(6)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(5)] += j_temp * 2.0 * 1.7007339999999999e+01;
  jac[INDEX(7)] += -j_temp * 3.3006740000000001e+01;

  //partial of rxn 17 wrt species
  if (T <= 1000.0) {
    Kc = (-4.6600633630000008e+00 + 1.1822328500000001e+00 * logT + T * (-2.6757265352666572e-05 + T * (-1.9870687130006796e-06 + T * (1.3761768948082653e-09 + -3.2820109195361345e-13 * T))) - -1.8538306819999998e+04 / T);
  } else {
    Kc = (5.9464085339999997e+00 + -3.3143557000000001e-01 * logT + T * (-5.7148033745785135e-04 + T * (1.4777809830730085e-07 + T * (-2.4178223038737304e-11 + 1.7136664100901320e-15 * T))) - -1.7868202613000001e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(2.5154082635789724e+01 - (3.1954442368556835e+02 / T));
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(15)] +=  1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  -1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(25)] +=  3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(27)] +=  -3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(35)] +=  2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  -2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(45)] +=  1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(47)] +=  -1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01 - kr * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(55)] +=  2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01 - kr * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(57)] +=  -1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01 - kr * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(65)] +=  1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] +=  -1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(75)] +=  1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(77)] +=  -(j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(85)] +=  (j_temp * -2.1422543718238485e-01);
  jac[INDEX(87)] +=  -9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  -8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(18)];
  //partial of rxn 18 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(18)]) * (2.0000000000000000e+00 + (2.6167417372676464e+03 / T)) + shared_temp[threadIdx.x + 3 * blockDim.x] * -1.0 - rev_rates[INDEX(18)] * (-1.0 + -T * (dBdT[INDEX(0)] + dBdT[INDEX(6)] - dBdT[INDEX(1)] - dBdT[INDEX(7)])))) * rho_inv;
  jac[INDEX(1)] += j_temp * 2.0158800000000001e+00;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(7)] += j_temp * 3.3006740000000001e+01;
  jac[INDEX(8)] += -j_temp * 3.4014679999999998e+01;

  //partial of rxn 18 wrt species
  if (T <= 1000.0) {
    Kc = (1.5412883609999999e+00 + -1.2998355999999989e-01 * logT + T * (1.8871113281473330e-03 + T * (-2.5089053330006806e-06 + T * (1.4548251081415985e-09 + -3.3542049995361335e-13 * T))) - -8.3942049330000009e+03 / T);
  } else {
    Kc = (-2.5788821060000005e+00 + 6.8948724000000050e-01 * logT + T * (-1.3589496300078514e-03 + T * (2.9453181030730083e-07 + T * (-3.6375500438737303e-11 + 1.9012757050901321e-15 * T))) - -8.4501744089999956e+03 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 2.0 * rev_rates[INDEX(18)]);
  kf = exp(9.4009607315848331e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  kr = kf / Kc;
  jac[INDEX(11)] +=  (j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(17)] +=  1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(18)] +=  -1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(21)] +=  2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(27)] +=  3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(28)] +=  -3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(31)] +=  1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(38)] +=  -2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(41)] +=  6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(47)] +=  1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(48)] +=  -1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(51)] +=  1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(57)] +=  1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(58)] +=  -2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(61)] +=  1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] +=  1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(68)] +=  -1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(71)] +=  6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(0)]);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(0)]);
  jac[INDEX(77)] +=  (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(0)]);
  jac[INDEX(78)] +=  -1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(0)]);
  jac[INDEX(81)] +=  5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(87)] +=  9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(88)] +=  -(j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(91)] +=  5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(98)] +=  -8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x] = fwd_rates[INDEX(19)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(19)];
  //partial of rxn 19 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * ((1.8115904334929860e+03 / T)) + shared_temp[threadIdx.x] * -1.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(5)] + dBdT[INDEX(4)] - dBdT[INDEX(1)] - dBdT[INDEX(7)])))) * rho_inv;
  jac[INDEX(2)] += -j_temp * 1.0079400000000001e+00;
  jac[INDEX(5)] += j_temp * 1.7007339999999999e+01;
  jac[INDEX(6)] += j_temp * 1.8015280000000001e+01;
  jac[INDEX(8)] += -j_temp * 3.4014679999999998e+01;

  //partial of rxn 19 wrt species
  if (T <= 1000.0) {
    Kc = (-5.3558688530000005e+00 + 1.4145433000000001e+00 * logT + T * (-1.9474646018526665e-03 + T * (-9.3253826300068034e-07 + T * (1.0173314456415985e-09 + -2.7442253795361340e-13 * T))) - -3.4449723939999996e+04 / T);
  } else {
    Kc = (7.5121151939999979e+00 + -5.3812269999999929e-01 * logT + T * (-1.0914845804578513e-03 + T * (3.1063749064063419e-07 + T * (-4.6347843655403966e-11 + 2.8676083400901325e-15 * T))) - -3.3757512300000002e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(2.3025850929940457e+01 - (1.8115904334929860e+03 / T));
  kr = kf / Kc;
  jac[INDEX(12)] +=  -5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(15)] +=  8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(16)] +=  8.9366827390519283e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(18)] +=  -1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(22)] +=  -(j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(25)] +=  1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(26)] +=  1.7873365478103857e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(28)] +=  -3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(32)] +=  -6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(35)] +=  1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(36)] +=  1.1259972248959336e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(38)] +=  -2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(42)] +=  -3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(45)] +=  5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(46)] +=  5.6299861244796678e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(48)] +=  -1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(52)] +=  -5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(55)] +=  (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(56)] +=  1.0592649997001296e+00 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(58)] +=  -2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(62)] +=  -5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01 - kr * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(65)] +=  9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01 - kr * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(66)] +=  (j_temp * 3.5690674632355568e-01 - kr * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(68)] +=  -1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01 - kr * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(72)] +=  -3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(75)] +=  5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(76)] +=  5.4580609899675037e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(78)] +=  -1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(82)] +=  -2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(85)] +=  5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(86)] +=  5.2963249985006478e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(88)] +=  -(j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(92)] +=  -2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(96)] +=  4.5096825873635726e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(98)] +=  -8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(20)];
  //partial of rxn 20 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(20)]) * (1.5100000000000000e+00 + (1.7260431074669282e+03 / T)) + shared_temp[threadIdx.x + 3 * blockDim.x] * -1.0 - rev_rates[INDEX(20)] * (-1.0 + -T * (dBdT[INDEX(1)] + dBdT[INDEX(5)] - dBdT[INDEX(0)] - dBdT[INDEX(4)])))) * rho_inv;
  jac[INDEX(1)] += -j_temp * 2.0158800000000001e+00;
  jac[INDEX(2)] += j_temp * 1.0079400000000001e+00;
  jac[INDEX(5)] += -j_temp * 1.7007339999999999e+01;
  jac[INDEX(6)] += j_temp * 1.8015280000000001e+01;

  //partial of rxn 20 wrt species
  if (T <= 1000.0) {
    Kc = (-2.2370938509999996e+00 + 3.6229401000000028e-01 * logT + T * (-3.8078186646473330e-03 + T * (3.5634357830006797e-06 + T * (-1.8136705573082654e-09 + 3.8919905395361339e-13 * T))) - -7.5172121869999974e+03 / T);
  } else {
    Kc = (4.1445887660000000e+00 + -8.9617436999999978e-01 * logT + T * (8.3894538700785145e-04 + T * (-1.3167241797396752e-07 + T * (1.4205879822070641e-11 + -7.4733377509013206e-16 * T))) - -7.4391352780000016e+03 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 2.0 * rev_rates[INDEX(20)]);
  kf = exp(1.2283033686666302e+01 + 1.51 * logT - (1.7260431074669282e+03 / T));
  kr = kf / Kc;
  jac[INDEX(11)] +=  -(j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(12)] +=  5.0000000000000000e-01 * (j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(15)] +=  -8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(16)] +=  8.9366827390519283e+00 * (j_temp * 9.2803892982949643e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(21)] +=  -2.0000000000000000e+00 * (j_temp * 9.6401946491474821e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(22)] +=  (j_temp * 9.6401946491474821e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(25)] +=  -1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(26)] +=  1.7873365478103857e+01 * (j_temp * 9.6401946491474821e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(31)] +=  -1.2599722489593360e-01 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(32)] +=  6.2998612447966798e-02 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(35)] +=  -1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(36)] +=  1.1259972248959336e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(41)] +=  -6.2998612447966798e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(42)] +=  3.1499306223983399e-02 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(45)] +=  -5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(46)] +=  5.6299861244796678e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(51)] +=  -1.1852999940025896e-01 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(0)]);
  jac[INDEX(52)] +=  5.9264999700129481e-02 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(0)]);
  jac[INDEX(55)] +=  -(j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(0)]);
  jac[INDEX(56)] +=  1.0592649997001296e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(0)]);
  jac[INDEX(61)] +=  -1.1189834407236524e-01 * (j_temp * 3.5690674632355568e-01 - kr * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(62)] +=  5.5949172036182618e-02 * (j_temp * 3.5690674632355568e-01 - kr * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(65)] +=  -9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01 - kr * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(66)] +=  (j_temp * 3.5690674632355568e-01 - kr * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(71)] +=  -6.1074798662333819e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(72)] +=  3.0537399331166910e-02 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(75)] +=  -5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(76)] +=  5.4580609899675037e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(81)] +=  -5.9264999700129481e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(82)] +=  2.9632499850064740e-02 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(85)] +=  -5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(86)] +=  5.2963249985006478e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(91)] +=  -5.0462601381796340e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(92)] +=  2.5231300690898170e-02 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  -4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(96)] +=  4.5096825873635726e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x] = pres_mod[INDEX(5)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(21)];
  //partial of rxn 21 wrt T
  conc_temp = (m - 0.30000000000000004 * conc[INDEX(8)] + 1.0 * conc[INDEX(0)] + 5.0 * conc[INDEX(5)]);
  Pr = conc_temp * (exp(3.4365993087130713e+00 - 0.53 * logT - (-8.5547326026057669e+02 / T)));
  Fcent = 2.6539999999999997e-01 * exp(T / -9.4000000000000000e+01) + 7.3460000000000003e-01 * exp(T / -1.7560000000000000e+03) + exp(-5.1820000000000000e+03 / T);
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * log10(fmax(Fcent, 1.0e-300)) - 0.4;
  B = 0.806 - 1.1762 * log10(fmax(Fcent, 1.0e-300)) - 0.14 * log10(fmax(Pr, 1.0e-300));
  lnF_AB = 2.0 * log(fmax(Fcent, 1.0e-300)) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)));
  j_temp = (shared_temp[threadIdx.x] * (((-5.3000e-01 + (-8.5547326026057669e+02 / T) - 1.0) / (T * (1.0 + Pr))) + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-2.9097730287517870e-01 * B + 5.1081716961460466e-01 * A) / Fcent) * (-2.8234042553191485e-03 * exp(T / -9.4000000000000000e+01) - 4.1833712984054670e-04 * exp(T / -1.7560000000000000e+03) + (5.1820000000000000e+03 / (T * T)) * exp(-5.1820000000000000e+03 / T))) - lnF_AB * (4.3429448190325176e-01 * B + 6.0801227466455256e-02 * A) * (-5.3000000000000003e-01 + (-8.5547326026057669e+02 / T) - 1.0) / T) * (shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(21)]) + (shared_temp[threadIdx.x] / T) * ((shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(21)]) * (-3.7000000000000000e-01) + shared_temp[threadIdx.x + 3 * blockDim.x] * -1.0 - rev_rates[INDEX(21)] * (-T * (dBdT[INDEX(7)] - 2.0 * dBdT[INDEX(4)])))) * rho_inv;
  jac[INDEX(5)] += j_temp * -2.0 * 1.7007339999999999e+01;
  jac[INDEX(8)] += j_temp * 3.4014679999999998e+01;

  //partial of rxn 21 wrt species
  if (T <= 1000.0) {
    Kc = (7.3508198260000004e+00 + -2.7079181700000001e+00 * logT + T * (2.1299063115000003e-03 + T * (1.2496155466666669e-06 + T * (-1.1512345533333333e-09 + 2.9481571150000003e-13 * T))) - -2.4932743220000000e+04 / T);
  } else {
    Kc = (-4.0164630899999985e+00 + -1.0207724900000006e+00 * logT + T * (1.9057287540000000e-03 + T * (-3.5906711766666662e-07 + T * (4.5589858100000000e-11 + -2.6136652849999998e-15 * T))) - -2.5579101699999999e+04 / T);
  }

  Kc = 8.2057360967184814e-02 * exp(Kc);
  pres_mod_temp = ((1.0 / (1.0 + Pr)) - log(fmax(Fcent, 1.0e-300)) * 2.0 * A * (B * 0.4342944819032518 + A * 0.06080122746645526) / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))) * (shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(21)]);
  j_temp = -mw_avg * rho_inv * shared_temp[threadIdx.x] * (2.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(21)] + pres_mod_temp);
  pres_mod_temp *= exp(3.4365993087130713e+00 - 0.53 * logT - (-8.5547326026057669e+02 / T)) * pow(Fcent, 1.0 / (1 + A * A / (B * B))) / (1.0 + Pr);
  kf = exp(2.5027330930150580e+01 - 0.37 * logT);
  kr = kf / Kc;
  jac[INDEX(15)] +=  -1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01 + 1.9280389298294964e+00 * pres_mod_temp);
  jac[INDEX(18)] +=  1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01 + 1.9280389298294964e+00 * pres_mod_temp);
  jac[INDEX(25)] +=  -3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp);
  jac[INDEX(28)] +=  3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01 + 9.6401946491474821e-01 * pres_mod_temp);
  jac[INDEX(35)] +=  -2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp);
  jac[INDEX(38)] +=  2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01 + 4.2886781649405936e-01 * pres_mod_temp);
  jac[INDEX(45)] +=  -1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(48)] +=  1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01 + -1.4226436701188128e-01 * pres_mod_temp);
  jac[INDEX(55)] +=  -2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp + shared_temp[threadIdx.x] * (kf * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]));
  jac[INDEX(58)] +=  2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01 + 3.9288728140880758e-01 * pres_mod_temp + shared_temp[threadIdx.x] * (kf * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]));
  jac[INDEX(65)] +=  -1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01 + 5.3569067463235553e+00 * pres_mod_temp);
  jac[INDEX(68)] +=  1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01 + 5.3569067463235553e+00 * pres_mod_temp);
  jac[INDEX(75)] +=  -1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(78)] +=  1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01 + -1.7824490209713328e-01 * pres_mod_temp);
  jac[INDEX(85)] +=  -(j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp + shared_temp[threadIdx.x] * (-kr));
  jac[INDEX(88)] +=  (j_temp * -2.1422543718238485e-01 + -2.1422543718238485e-01 * pres_mod_temp + shared_temp[threadIdx.x] * (-kr));
  jac[INDEX(95)] +=  -8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01 + -7.2602775520927776e-01 * pres_mod_temp);
  jac[INDEX(98)] +=  8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01 + -7.2602775520927776e-01 * pres_mod_temp);

  shared_temp[threadIdx.x + 2 * blockDim.x] = fwd_rates[INDEX(22)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(22)];
  //partial of rxn 22 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 2 * blockDim.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * (2.3999999999999999e+00 + (-1.0617932818528334e+03 / T)) + shared_temp[threadIdx.x + 2 * blockDim.x] * -1.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(5)] + dBdT[INDEX(2)] - 2.0 * dBdT[INDEX(4)])))) * rho_inv;
  jac[INDEX(3)] += j_temp * 1.5999400000000000e+01;
  jac[INDEX(5)] += j_temp * -2.0 * 1.7007339999999999e+01;
  jac[INDEX(6)] += j_temp * 1.8015280000000001e+01;

  //partial of rxn 22 wrt species
  if (T <= 1000.0) {
    Kc = (2.0278753680000001e+00 + -6.1712319999999954e-01 * logT + T * (-2.5655894999999994e-04 + T * (6.5459820833333341e-07 + T * (-3.2114751666666664e-10 + 5.7820423999999999e-14 * T))) - -8.4016286199999995e+03 / T);
  } else {
    Kc = (1.3800786099999995e+00 + -5.8236206999999984e-01 * logT + T * (4.9704224715000007e-04 + T * (-6.2522419183333327e-08 + T * (5.7360453583333335e-12 + -2.7170195449999994e-16 * T))) - -8.5040320000000029e+03 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(3.5751506887855933e+00 + 2.4 * logT - (-1.0617932818528334e+03 / T));
  kr = kf / Kc;
  jac[INDEX(13)] +=  7.9366827390519274e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(15)] +=  -1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(16)] +=  8.9366827390519283e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(23)] +=  1.5873365478103855e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(25)] +=  -3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(26)] +=  1.7873365478103857e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(33)] +=  (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(35)] +=  -2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(36)] +=  1.1259972248959336e+00 * (j_temp * 4.2886781649405936e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(43)] +=  5.0000000000000000e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(45)] +=  -1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(46)] +=  5.6299861244796678e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(53)] +=  9.4073500029987056e-01 * (j_temp * 3.9288728140880758e-01 + kf * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(55)] +=  -2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01 + kf * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(56)] +=  1.0592649997001296e+00 * (j_temp * 3.9288728140880758e-01 + kf * 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(63)] +=  8.8810165592763468e-01 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(65)] +=  -1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(66)] +=  (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(2)]);
  jac[INDEX(73)] +=  4.8473130033441653e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(75)] +=  -1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(76)] +=  5.4580609899675037e-01 * (j_temp * -1.7824490209713328e-01);
  jac[INDEX(83)] +=  4.7036750014993528e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(85)] +=  -(j_temp * -2.1422543718238485e-01);
  jac[INDEX(86)] +=  5.2963249985006478e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(93)] +=  4.0050565735456090e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  -8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(96)] +=  4.5096825873635726e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x] = fwd_rates[INDEX(23)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = rev_rates[INDEX(23)];
  //partial of rxn 23 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x] - shared_temp[threadIdx.x + 2 * blockDim.x]) * ((-2.5160978242958137e+02 / T)) + shared_temp[threadIdx.x] * -1.0 - shared_temp[threadIdx.x + 2 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(5)] + dBdT[INDEX(3)] - dBdT[INDEX(6)] - dBdT[INDEX(4)])))) * rho_inv;
  jac[INDEX(4)] += j_temp * 3.1998799999999999e+01;
  jac[INDEX(5)] += -j_temp * 1.7007339999999999e+01;
  jac[INDEX(6)] += j_temp * 1.8015280000000001e+01;
  jac[INDEX(7)] += -j_temp * 3.3006740000000001e+01;

  //partial of rxn 23 wrt species
  if (T <= 1000.0) {
    Kc = (-4.9137694999999981e-01 + -3.1271651999999950e-01 * logT + T * (1.0586348849999999e-03 + T * (-1.5680872316666664e-06 + T * (1.0823547516666667e-09 + -2.8203297000000007e-13 * T))) - -3.5267558860000005e+04 / T);
  } else {
    Kc = (2.9517713799999998e+00 + -7.9356824000000037e-01 * logT + T * (4.3587786700000010e-04 + T * (-6.9147710833333326e-08 + T * (7.1773627999999992e-12 + -2.9010321499999993e-16 * T))) - -3.5063268533000002e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x] - 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]);
  kf = exp(2.3397414486372941e+01 - (-2.5160978242958137e+02 / T));
  kr = kf / Kc;
  jac[INDEX(14)] +=  1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(15)] +=  -8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(16)] +=  8.9366827390519283e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  -1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(24)] +=  3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(25)] +=  -1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(26)] +=  1.7873365478103857e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(27)] +=  -3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(34)] +=  2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(35)] +=  -1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(36)] +=  1.1259972248959336e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  -2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(44)] +=  (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(45)] +=  -5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(46)] +=  5.6299861244796678e-01 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(47)] +=  -1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(54)] +=  1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(55)] +=  -(j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(56)] +=  1.0592649997001296e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(57)] +=  -1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(6)]);
  jac[INDEX(64)] +=  1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(65)] +=  -9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(66)] +=  (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(67)] +=  -1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(74)] +=  9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(75)] +=  -5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(76)] +=  5.4580609899675037e-01 * (j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(77)] +=  -(j_temp * -1.7824490209713328e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(84)] +=  9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(85)] +=  -5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(86)] +=  5.2963249985006478e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(87)] +=  -9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(94)] +=  8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  -4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(96)] +=  4.5096825873635726e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  -8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(24)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = rev_rates[INDEX(24)];
  //partial of rxn 24 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 3 * blockDim.x] - shared_temp[threadIdx.x + 2 * blockDim.x]) * ((2.1487475419486250e+02 / T)) + shared_temp[threadIdx.x + 3 * blockDim.x] * -1.0 - shared_temp[threadIdx.x + 2 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(5)] + dBdT[INDEX(6)] - dBdT[INDEX(7)] - dBdT[INDEX(4)])))) * rho_inv;
  jac[INDEX(5)] += -j_temp * 1.7007339999999999e+01;
  jac[INDEX(6)] += j_temp * 1.8015280000000001e+01;
  jac[INDEX(7)] += j_temp * 3.3006740000000001e+01;
  jac[INDEX(8)] += -j_temp * 3.4014679999999998e+01;

  //partial of rxn 24 wrt species
  if (T <= 1000.0) {
    Kc = (-6.9580548999999969e-01 + 2.3231045000000039e-01 * logT + T * (-1.9207073365000000e-03 + T * (1.0545304499999991e-06 + T * (-3.5884544916666708e-10 + 5.3778554000000095e-14 * T))) - -1.5911417119999998e+04 / T);
  } else {
    Kc = (1.5657066599999991e+00 + -2.0668712999999928e-01 * logT + T * (-5.2000424300000002e-04 + T * (1.6285939233333333e-07 + T * (-2.2169620616666662e-11 + 1.1539419300000000e-15 * T))) - -1.5889309687000001e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]);
  kf = exp(2.1416413017506358e+01 - (2.1487475419486250e+02 / T));
  kr = kf / Kc;
  jac[INDEX(15)] +=  -8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(16)] +=  8.9366827390519283e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(18)] +=  -1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(25)] +=  -1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(26)] +=  1.7873365478103857e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(27)] +=  3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(28)] +=  -3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(35)] +=  -1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(36)] +=  1.1259972248959336e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(38)] +=  -2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(45)] +=  -5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(46)] +=  5.6299861244796678e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(47)] +=  1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(48)] +=  -1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(55)] +=  -(j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(56)] +=  1.0592649997001296e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(57)] +=  1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(58)] +=  -2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(65)] +=  -9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(66)] +=  (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(67)] +=  1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(68)] +=  -1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(75)] +=  -5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(76)] +=  5.4580609899675037e-01 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(77)] +=  (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(78)] +=  -1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(85)] +=  -5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(86)] +=  5.2963249985006478e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(87)] +=  9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(88)] +=  -(j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(95)] +=  -4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(96)] +=  4.5096825873635726e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(98)] +=  -8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x] = fwd_rates[INDEX(25)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = rev_rates[INDEX(25)];
  //partial of rxn 25 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x] - shared_temp[threadIdx.x + 3 * blockDim.x]) * ((1.4799687402507978e+04 / T)) + shared_temp[threadIdx.x] * -1.0 - shared_temp[threadIdx.x + 3 * blockDim.x] * (-1.0 + -T * (dBdT[INDEX(5)] + dBdT[INDEX(6)] - dBdT[INDEX(7)] - dBdT[INDEX(4)])))) * rho_inv;
  jac[INDEX(5)] += -j_temp * 1.7007339999999999e+01;
  jac[INDEX(6)] += j_temp * 1.8015280000000001e+01;
  jac[INDEX(7)] += j_temp * 3.3006740000000001e+01;
  jac[INDEX(8)] += -j_temp * 3.4014679999999998e+01;

  //partial of rxn 25 wrt species
  if (T <= 1000.0) {
    Kc = (-6.9580548999999969e-01 + 2.3231045000000039e-01 * logT + T * (-1.9207073365000000e-03 + T * (1.0545304499999991e-06 + T * (-3.5884544916666708e-10 + 5.3778554000000095e-14 * T))) - -1.5911417119999998e+04 / T);
  } else {
    Kc = (1.5657066599999991e+00 + -2.0668712999999928e-01 * logT + T * (-5.2000424300000002e-04 + T * (1.6285939233333333e-07 + T * (-2.2169620616666662e-11 + 1.1539419300000000e-15 * T))) - -1.5889309687000001e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x] - 2.0 * shared_temp[threadIdx.x + 3 * blockDim.x]);
  kf = exp(3.5069404645972853e+01 - (1.4799687402507978e+04 / T));
  kr = kf / Kc;
  jac[INDEX(15)] +=  -8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(16)] +=  8.9366827390519283e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(18)] +=  -1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(25)] +=  -1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(26)] +=  1.7873365478103857e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(27)] +=  3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(28)] +=  -3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(35)] +=  -1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(36)] +=  1.1259972248959336e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(38)] +=  -2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(45)] +=  -5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(46)] +=  5.6299861244796678e-01 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(47)] +=  1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(48)] +=  -1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01);
  jac[INDEX(55)] +=  -(j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(56)] +=  1.0592649997001296e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(57)] +=  1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(58)] +=  -2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01 + kf * conc[INDEX(7)]);
  jac[INDEX(65)] +=  -9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(66)] +=  (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(67)] +=  1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(68)] +=  -1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(6)]);
  jac[INDEX(75)] +=  -5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(76)] +=  5.4580609899675037e-01 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(77)] +=  (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(78)] +=  -1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(85)] +=  -5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(86)] +=  5.2963249985006478e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(87)] +=  9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(88)] +=  -(j_temp * -2.1422543718238485e-01 + kf * shared_temp[threadIdx.x + 1 * blockDim.x]);
  jac[INDEX(95)] +=  -4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(96)] +=  4.5096825873635726e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(98)] +=  -8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 2 * blockDim.x] = conc[INDEX(6)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(26)];
  //partial of rxn 26 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(26)]) * ((-8.2024789072043529e+02 / T)) + shared_temp[threadIdx.x + 3 * blockDim.x] * -1.0 - rev_rates[INDEX(26)] * (-1.0 + -T * (dBdT[INDEX(7)] + dBdT[INDEX(3)] - 2.0 * dBdT[INDEX(6)])))) * rho_inv;
  jac[INDEX(4)] += j_temp * 3.1998799999999999e+01;
  jac[INDEX(7)] += j_temp * -2.0 * 3.3006740000000001e+01;
  jac[INDEX(8)] += j_temp * 3.4014679999999998e+01;

  //partial of rxn 26 wrt species
  if (T <= 1000.0) {
    Kc = (2.0442853999999988e-01 + -5.4502696999999989e-01 * logT + T * (2.9793422214999996e-03 + T * (-2.6226176816666659e-06 + T * (1.4412002008333338e-09 + -3.3581152400000011e-13 * T))) - -1.9356141739999999e+04 / T);
  } else {
    Kc = (1.3860647200000007e+00 + -5.8688111000000109e-01 * logT + T * (9.5588210999999991e-04 + T * (-2.3200710316666665e-07 + T * (2.9346983416666660e-11 + -1.4440451450000000e-15 * T))) - -1.9173958846000001e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 2.0 * rev_rates[INDEX(26)]);
  kf = exp(1.8683045008419857e+01 - (-8.2024789072043529e+02 / T));
  kr = kf / Kc;
  jac[INDEX(14)] +=  1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  -3.2746730956207706e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(18)] +=  1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(24)] +=  3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(27)] +=  -6.5493461912415412e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(28)] +=  3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(34)] +=  2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  -4.1259972248959338e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(38)] +=  2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(44)] +=  (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(7)]);
  jac[INDEX(47)] +=  -2.0629986124479669e+00 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(7)]);
  jac[INDEX(48)] +=  1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(7)]);
  jac[INDEX(54)] +=  1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(57)] +=  -3.8814700005997413e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(58)] +=  2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(64)] +=  1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] +=  -3.6643049677829040e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(68)] +=  1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(74)] +=  9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(77)] +=  -2.0000000000000000e+00 * (j_temp * -1.7824490209713328e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(78)] +=  1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(84)] +=  9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(87)] +=  -1.9407350002998707e+00 * (j_temp * -2.1422543718238485e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(88)] +=  (j_temp * -2.1422543718238485e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(94)] +=  8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  -1.6524852308000402e+00 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(98)] +=  8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 1 * blockDim.x] = fwd_rates[INDEX(27)];
  shared_temp[threadIdx.x] = rev_rates[INDEX(27)];
  //partial of rxn 27 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 1 * blockDim.x] - shared_temp[threadIdx.x]) * ((6.0386347783099527e+03 / T)) + shared_temp[threadIdx.x + 1 * blockDim.x] * -1.0 - shared_temp[threadIdx.x] * (-1.0 + -T * (dBdT[INDEX(7)] + dBdT[INDEX(3)] - 2.0 * dBdT[INDEX(6)])))) * rho_inv;
  jac[INDEX(4)] += j_temp * 3.1998799999999999e+01;
  jac[INDEX(7)] += j_temp * -2.0 * 3.3006740000000001e+01;
  jac[INDEX(8)] += j_temp * 3.4014679999999998e+01;

  //partial of rxn 27 wrt species
  if (T <= 1000.0) {
    Kc = (2.0442853999999988e-01 + -5.4502696999999989e-01 * logT + T * (2.9793422214999996e-03 + T * (-2.6226176816666659e-06 + T * (1.4412002008333338e-09 + -3.3581152400000011e-13 * T))) - -1.9356141739999999e+04 / T);
  } else {
    Kc = (1.3860647200000007e+00 + -5.8688111000000109e-01 * logT + T * (9.5588210999999991e-04 + T * (-2.3200710316666665e-07 + T * (2.9346983416666660e-11 + -1.4440451450000000e-15 * T))) - -1.9173958846000001e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 2.0 * shared_temp[threadIdx.x]);
  kf = exp(2.6763520548223827e+01 - (6.0386347783099527e+03 / T));
  kr = kf / Kc;
  jac[INDEX(14)] +=  1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  -3.2746730956207706e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(18)] +=  1.6873365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(24)] +=  3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(27)] +=  -6.5493461912415412e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(28)] +=  3.3746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(34)] +=  2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  -4.1259972248959338e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(38)] +=  2.1259972248959333e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(44)] +=  (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(7)]);
  jac[INDEX(47)] +=  -2.0629986124479669e+00 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(7)]);
  jac[INDEX(48)] +=  1.0629986124479667e+00 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(7)]);
  jac[INDEX(54)] +=  1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(57)] +=  -3.8814700005997413e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(58)] +=  2.0000000000000000e+00 * (j_temp * 3.9288728140880758e-01);
  jac[INDEX(64)] +=  1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(67)] +=  -3.6643049677829040e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(68)] +=  1.8881016559276347e+00 * (j_temp * 3.5690674632355568e-01);
  jac[INDEX(74)] +=  9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(77)] +=  -2.0000000000000000e+00 * (j_temp * -1.7824490209713328e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(78)] +=  1.0305373993311668e+00 * (j_temp * -1.7824490209713328e-01 + kf * 2.0 * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(84)] +=  9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(87)] +=  -1.9407350002998707e+00 * (j_temp * -2.1422543718238485e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(88)] +=  (j_temp * -2.1422543718238485e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(94)] +=  8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  -1.6524852308000402e+00 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(98)] +=  8.5147391609091816e-01 * (j_temp * -4.2602775520927771e-01);

  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(28)];
  //partial of rxn 28 wrt T
  j_temp = ((1.0 / T) * ((shared_temp[threadIdx.x + 3 * blockDim.x] - rev_rates[INDEX(28)]) * ((8.7207950590092896e+03 / T)) + shared_temp[threadIdx.x + 3 * blockDim.x] * -1.0 - rev_rates[INDEX(28)] * (-1.0 + -T * (dBdT[INDEX(5)] + dBdT[INDEX(3)] - dBdT[INDEX(6)] - dBdT[INDEX(4)])))) * rho_inv;
  jac[INDEX(4)] += j_temp * 3.1998799999999999e+01;
  jac[INDEX(5)] += -j_temp * 1.7007339999999999e+01;
  jac[INDEX(6)] += j_temp * 1.8015280000000001e+01;
  jac[INDEX(7)] += -j_temp * 3.3006740000000001e+01;

  //partial of rxn 28 wrt species
  if (T <= 1000.0) {
    Kc = (-4.9137694999999981e-01 + -3.1271651999999950e-01 * logT + T * (1.0586348849999999e-03 + T * (-1.5680872316666664e-06 + T * (1.0823547516666667e-09 + -2.8203297000000007e-13 * T))) - -3.5267558860000005e+04 / T);
  } else {
    Kc = (2.9517713799999998e+00 + -7.9356824000000037e-01 * logT + T * (4.3587786700000010e-04 + T * (-6.9147710833333326e-08 + T * (7.1773627999999992e-12 + -2.9010321499999993e-16 * T))) - -3.5063268533000002e+04 / T);
  }

  Kc = exp(Kc);
  j_temp = -mw_avg * rho_inv * (2.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 2.0 * rev_rates[INDEX(28)]);
  kf = exp(2.9240459028362650e+01 - (8.7207950590092896e+03 / T));
  kr = kf / Kc;
  jac[INDEX(14)] +=  1.5873365478103855e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(15)] +=  -8.4366827390519266e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(16)] +=  8.9366827390519283e+00 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(17)] +=  -1.6373365478103853e+01 * (j_temp * 9.2803892982949643e-01);
  jac[INDEX(24)] +=  3.1746730956207710e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(25)] +=  -1.6873365478103853e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(26)] +=  1.7873365478103857e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(27)] +=  -3.2746730956207706e+01 * (j_temp * 9.6401946491474821e-01);
  jac[INDEX(34)] +=  2.0000000000000000e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(35)] +=  -1.0629986124479667e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(36)] +=  1.1259972248959336e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(37)] +=  -2.0629986124479669e+00 * (j_temp * 4.2886781649405936e-01);
  jac[INDEX(44)] +=  (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(45)] +=  -5.3149930622398334e-01 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(46)] +=  5.6299861244796678e-01 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(47)] +=  -1.0314993062239834e+00 * (j_temp * -1.4226436701188128e-01 - kr * conc[INDEX(5)]);
  jac[INDEX(54)] +=  1.8814700005997411e+00 * (j_temp * 3.9288728140880758e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(55)] +=  -(j_temp * 3.9288728140880758e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(56)] +=  1.0592649997001296e+00 * (j_temp * 3.9288728140880758e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(57)] +=  -1.9407350002998707e+00 * (j_temp * 3.9288728140880758e-01 + kf * shared_temp[threadIdx.x + 2 * blockDim.x]);
  jac[INDEX(64)] +=  1.7762033118552694e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(65)] +=  -9.4405082796381734e-01 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(66)] +=  (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(67)] +=  -1.8321524838914520e+00 * (j_temp * 3.5690674632355568e-01 - kr * conc[INDEX(3)]);
  jac[INDEX(74)] +=  9.6946260066883305e-01 * (j_temp * -1.7824490209713328e-01 + kf * conc[INDEX(4)]);
  jac[INDEX(75)] +=  -5.1526869966558342e-01 * (j_temp * -1.7824490209713328e-01 + kf * conc[INDEX(4)]);
  jac[INDEX(76)] +=  5.4580609899675037e-01 * (j_temp * -1.7824490209713328e-01 + kf * conc[INDEX(4)]);
  jac[INDEX(77)] +=  -(j_temp * -1.7824490209713328e-01 + kf * conc[INDEX(4)]);
  jac[INDEX(84)] +=  9.4073500029987056e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(85)] +=  -5.0000000000000000e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(86)] +=  5.2963249985006478e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(87)] +=  -9.7036750014993534e-01 * (j_temp * -2.1422543718238485e-01);
  jac[INDEX(94)] +=  8.0101131470912179e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(95)] +=  -4.2573695804545908e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(96)] +=  4.5096825873635726e-01 * (j_temp * -4.2602775520927771e-01);
  jac[INDEX(97)] +=  -8.2624261540002009e-01 * (j_temp * -4.2602775520927771e-01);

  // species enthalpies
  double * __restrict__ h = d_mem->h;
  eval_h(T, h);

  // species specific heats
  double * __restrict__ cp = d_mem->cp;
  eval_cp(T, cp);

  // average specific heat
  register double cp_avg;
  cp_avg = (y[INDEX(1)] * cp[INDEX(0)]) + (y[INDEX(2)] * cp[INDEX(1)]) + 
     (y[INDEX(3)] * cp[INDEX(2)]) + (y[INDEX(4)] * cp[INDEX(3)]) + (y[INDEX(5)] * cp[INDEX(4)]) + 
     (y[INDEX(6)] * cp[INDEX(5)]) + (y[INDEX(7)] * cp[INDEX(6)]) + (y[INDEX(8)] * cp[INDEX(7)]) + 
     (y[INDEX(9)] * cp[INDEX(8)]) + (y_N * cp[INDEX(9)]);
  jac[INDEX(0)] = 0.0;
  register double working_temp = (1.0 / cp_avg);
  j_temp = 1.0 / (rho * cp_avg * cp_avg);
  //Finish dYk / Yj's
  //And dT/dYj's
  jac[INDEX(11)] += (spec_rates[INDEX(0)] * mw_avg * 9.2803892982949643e-01 * rho_inv);
  jac[INDEX(10)] = -(h[INDEX(0)] * (working_temp * jac[INDEX(11)] - (j_temp * (cp[INDEX(0)] - cp[INDEX(9)]) * spec_rates[INDEX(0)] * 2.01588000e+00)));
  jac[INDEX(21)] += (spec_rates[INDEX(0)] * mw_avg * 1.9280389298294964e+00 * rho_inv);
  jac[INDEX(20)] = -(h[INDEX(0)] * (working_temp * jac[INDEX(21)] - (j_temp * (cp[INDEX(1)] - cp[INDEX(9)]) * spec_rates[INDEX(0)] * 2.01588000e+00)));
  jac[INDEX(31)] += (spec_rates[INDEX(0)] * mw_avg * 5.4036154725429976e-02 * rho_inv);
  jac[INDEX(30)] = -(h[INDEX(0)] * (working_temp * jac[INDEX(31)] - (j_temp * (cp[INDEX(2)] - cp[INDEX(9)]) * spec_rates[INDEX(0)] * 2.01588000e+00)));
  jac[INDEX(41)] += (spec_rates[INDEX(0)] * mw_avg * -8.9624577225368204e-03 * rho_inv);
  jac[INDEX(40)] = -(h[INDEX(0)] * (working_temp * jac[INDEX(41)] - (j_temp * (cp[INDEX(3)] - cp[INDEX(9)]) * spec_rates[INDEX(0)] * 2.01588000e+00)));
  jac[INDEX(51)] += (spec_rates[INDEX(0)] * mw_avg * 4.6568929229755335e-02 * rho_inv);
  jac[INDEX(50)] = -(h[INDEX(0)] * (working_temp * jac[INDEX(51)] - (j_temp * (cp[INDEX(4)] - cp[INDEX(9)]) * spec_rates[INDEX(0)] * 2.01588000e+00)));
  jac[INDEX(61)] += (spec_rates[INDEX(0)] * mw_avg * 3.9937273901861609e-02 * rho_inv);
  jac[INDEX(60)] = -(h[INDEX(0)] * (working_temp * jac[INDEX(61)] - (j_temp * (cp[INDEX(5)] - cp[INDEX(9)]) * spec_rates[INDEX(0)] * 2.01588000e+00)));
  jac[INDEX(71)] += (spec_rates[INDEX(0)] * mw_avg * -1.0886271508169818e-02 * rho_inv);
  jac[INDEX(70)] = -(h[INDEX(0)] * (working_temp * jac[INDEX(71)] - (j_temp * (cp[INDEX(6)] - cp[INDEX(9)]) * spec_rates[INDEX(0)] * 2.01588000e+00)));
  jac[INDEX(81)] += (spec_rates[INDEX(0)] * mw_avg * -1.2696070470374144e-02 * rho_inv);
  jac[INDEX(80)] = -(h[INDEX(0)] * (working_temp * jac[INDEX(81)] - (j_temp * (cp[INDEX(7)] - cp[INDEX(9)]) * spec_rates[INDEX(0)] * 2.01588000e+00)));
  jac[INDEX(91)] += (spec_rates[INDEX(0)] * mw_avg * -2.1498468788707290e-02 * rho_inv);
  jac[INDEX(90)] = -(h[INDEX(0)] * (working_temp * jac[INDEX(91)] - (j_temp * (cp[INDEX(8)] - cp[INDEX(9)]) * spec_rates[INDEX(0)] * 2.01588000e+00)));
  jac[INDEX(12)] += (spec_rates[INDEX(1)] * mw_avg * 4.6401946491474821e-01 * rho_inv);
  jac[INDEX(10)] -= (h[INDEX(1)] * (working_temp * jac[INDEX(12)] - (j_temp * (cp[INDEX(0)] - cp[INDEX(9)]) * spec_rates[INDEX(1)] * 1.00794000e+00)));
  jac[INDEX(22)] += (spec_rates[INDEX(1)] * mw_avg * 9.6401946491474821e-01 * rho_inv);
  jac[INDEX(20)] -= (h[INDEX(1)] * (working_temp * jac[INDEX(22)] - (j_temp * (cp[INDEX(1)] - cp[INDEX(9)]) * spec_rates[INDEX(1)] * 1.00794000e+00)));
  jac[INDEX(32)] += (spec_rates[INDEX(1)] * mw_avg * 2.7018077362714988e-02 * rho_inv);
  jac[INDEX(30)] -= (h[INDEX(1)] * (working_temp * jac[INDEX(32)] - (j_temp * (cp[INDEX(2)] - cp[INDEX(9)]) * spec_rates[INDEX(1)] * 1.00794000e+00)));
  jac[INDEX(42)] += (spec_rates[INDEX(1)] * mw_avg * -4.4812288612684102e-03 * rho_inv);
  jac[INDEX(40)] -= (h[INDEX(1)] * (working_temp * jac[INDEX(42)] - (j_temp * (cp[INDEX(3)] - cp[INDEX(9)]) * spec_rates[INDEX(1)] * 1.00794000e+00)));
  jac[INDEX(52)] += (spec_rates[INDEX(1)] * mw_avg * 2.3284464614877667e-02 * rho_inv);
  jac[INDEX(50)] -= (h[INDEX(1)] * (working_temp * jac[INDEX(52)] - (j_temp * (cp[INDEX(4)] - cp[INDEX(9)]) * spec_rates[INDEX(1)] * 1.00794000e+00)));
  jac[INDEX(62)] += (spec_rates[INDEX(1)] * mw_avg * 1.9968636950930804e-02 * rho_inv);
  jac[INDEX(60)] -= (h[INDEX(1)] * (working_temp * jac[INDEX(62)] - (j_temp * (cp[INDEX(5)] - cp[INDEX(9)]) * spec_rates[INDEX(1)] * 1.00794000e+00)));
  jac[INDEX(72)] += (spec_rates[INDEX(1)] * mw_avg * -5.4431357540849091e-03 * rho_inv);
  jac[INDEX(70)] -= (h[INDEX(1)] * (working_temp * jac[INDEX(72)] - (j_temp * (cp[INDEX(6)] - cp[INDEX(9)]) * spec_rates[INDEX(1)] * 1.00794000e+00)));
  jac[INDEX(82)] += (spec_rates[INDEX(1)] * mw_avg * -6.3480352351870722e-03 * rho_inv);
  jac[INDEX(80)] -= (h[INDEX(1)] * (working_temp * jac[INDEX(82)] - (j_temp * (cp[INDEX(7)] - cp[INDEX(9)]) * spec_rates[INDEX(1)] * 1.00794000e+00)));
  jac[INDEX(92)] += (spec_rates[INDEX(1)] * mw_avg * -1.0749234394353645e-02 * rho_inv);
  jac[INDEX(90)] -= (h[INDEX(1)] * (working_temp * jac[INDEX(92)] - (j_temp * (cp[INDEX(8)] - cp[INDEX(9)]) * spec_rates[INDEX(1)] * 1.00794000e+00)));
  jac[INDEX(13)] += (spec_rates[INDEX(2)] * mw_avg * 7.3655505555459868e+00 * rho_inv);
  jac[INDEX(10)] -= (h[INDEX(2)] * (working_temp * jac[INDEX(13)] - (j_temp * (cp[INDEX(0)] - cp[INDEX(9)]) * spec_rates[INDEX(2)] * 1.59994000e+01)));
  jac[INDEX(23)] += (spec_rates[INDEX(2)] * mw_avg * 1.5302233294597915e+01 * rho_inv);
  jac[INDEX(20)] -= (h[INDEX(2)] * (working_temp * jac[INDEX(23)] - (j_temp * (cp[INDEX(1)] - cp[INDEX(9)]) * spec_rates[INDEX(2)] * 1.59994000e+01)));
  jac[INDEX(33)] += (spec_rates[INDEX(2)] * mw_avg * 4.2886781649405936e-01 * rho_inv);
  jac[INDEX(30)] -= (h[INDEX(2)] * (working_temp * jac[INDEX(33)] - (j_temp * (cp[INDEX(2)] - cp[INDEX(9)]) * spec_rates[INDEX(2)] * 1.59994000e+01)));
  jac[INDEX(43)] += (spec_rates[INDEX(2)] * mw_avg * -7.1132183505940638e-02 * rho_inv);
  jac[INDEX(40)] -= (h[INDEX(2)] * (working_temp * jac[INDEX(43)] - (j_temp * (cp[INDEX(3)] - cp[INDEX(9)]) * spec_rates[INDEX(2)] * 1.59994000e+01)));
  jac[INDEX(53)] += (spec_rates[INDEX(2)] * mw_avg * 3.6960281679392992e-01 * rho_inv);
  jac[INDEX(50)] -= (h[INDEX(2)] * (working_temp * jac[INDEX(53)] - (j_temp * (cp[INDEX(4)] - cp[INDEX(9)]) * spec_rates[INDEX(2)] * 1.59994000e+01)));
  jac[INDEX(63)] += (spec_rates[INDEX(2)] * mw_avg * 3.1696947242169404e-01 * rho_inv);
  jac[INDEX(60)] -= (h[INDEX(2)] * (working_temp * jac[INDEX(63)] - (j_temp * (cp[INDEX(5)] - cp[INDEX(9)]) * spec_rates[INDEX(2)] * 1.59994000e+01)));
  jac[INDEX(73)] += (spec_rates[INDEX(2)] * mw_avg * -8.6400883171524182e-02 * rho_inv);
  jac[INDEX(70)] -= (h[INDEX(2)] * (working_temp * jac[INDEX(73)] - (j_temp * (cp[INDEX(6)] - cp[INDEX(9)]) * spec_rates[INDEX(2)] * 1.59994000e+01)));
  jac[INDEX(83)] += (spec_rates[INDEX(2)] * mw_avg * -1.0076468335600536e-01 * rho_inv);
  jac[INDEX(80)] -= (h[INDEX(2)] * (working_temp * jac[INDEX(83)] - (j_temp * (cp[INDEX(7)] - cp[INDEX(9)]) * spec_rates[INDEX(2)] * 1.59994000e+01)));
  jac[INDEX(93)] += (spec_rates[INDEX(2)] * mw_avg * -1.7062652615137971e-01 * rho_inv);
  jac[INDEX(90)] -= (h[INDEX(2)] * (working_temp * jac[INDEX(93)] - (j_temp * (cp[INDEX(8)] - cp[INDEX(9)]) * spec_rates[INDEX(2)] * 1.59994000e+01)));
  jac[INDEX(14)] += (spec_rates[INDEX(3)] * mw_avg * 1.4731101111091974e+01 * rho_inv);
  jac[INDEX(10)] -= (h[INDEX(3)] * (working_temp * jac[INDEX(14)] - (j_temp * (cp[INDEX(0)] - cp[INDEX(9)]) * spec_rates[INDEX(3)] * 3.19988000e+01)));
  jac[INDEX(24)] += (spec_rates[INDEX(3)] * mw_avg * 3.0604466589195830e+01 * rho_inv);
  jac[INDEX(20)] -= (h[INDEX(3)] * (working_temp * jac[INDEX(24)] - (j_temp * (cp[INDEX(1)] - cp[INDEX(9)]) * spec_rates[INDEX(3)] * 3.19988000e+01)));
  jac[INDEX(34)] += (spec_rates[INDEX(3)] * mw_avg * 8.5773563298811872e-01 * rho_inv);
  jac[INDEX(30)] -= (h[INDEX(3)] * (working_temp * jac[INDEX(34)] - (j_temp * (cp[INDEX(2)] - cp[INDEX(9)]) * spec_rates[INDEX(3)] * 3.19988000e+01)));
  jac[INDEX(44)] += (spec_rates[INDEX(3)] * mw_avg * -1.4226436701188128e-01 * rho_inv);
  jac[INDEX(40)] -= (h[INDEX(3)] * (working_temp * jac[INDEX(44)] - (j_temp * (cp[INDEX(3)] - cp[INDEX(9)]) * spec_rates[INDEX(3)] * 3.19988000e+01)));
  jac[INDEX(54)] += (spec_rates[INDEX(3)] * mw_avg * 7.3920563358785984e-01 * rho_inv);
  jac[INDEX(50)] -= (h[INDEX(3)] * (working_temp * jac[INDEX(54)] - (j_temp * (cp[INDEX(4)] - cp[INDEX(9)]) * spec_rates[INDEX(3)] * 3.19988000e+01)));
  jac[INDEX(64)] += (spec_rates[INDEX(3)] * mw_avg * 6.3393894484338809e-01 * rho_inv);
  jac[INDEX(60)] -= (h[INDEX(3)] * (working_temp * jac[INDEX(64)] - (j_temp * (cp[INDEX(5)] - cp[INDEX(9)]) * spec_rates[INDEX(3)] * 3.19988000e+01)));
  jac[INDEX(74)] += (spec_rates[INDEX(3)] * mw_avg * -1.7280176634304836e-01 * rho_inv);
  jac[INDEX(70)] -= (h[INDEX(3)] * (working_temp * jac[INDEX(74)] - (j_temp * (cp[INDEX(6)] - cp[INDEX(9)]) * spec_rates[INDEX(3)] * 3.19988000e+01)));
  jac[INDEX(84)] += (spec_rates[INDEX(3)] * mw_avg * -2.0152936671201072e-01 * rho_inv);
  jac[INDEX(80)] -= (h[INDEX(3)] * (working_temp * jac[INDEX(84)] - (j_temp * (cp[INDEX(7)] - cp[INDEX(9)]) * spec_rates[INDEX(3)] * 3.19988000e+01)));
  jac[INDEX(94)] += (spec_rates[INDEX(3)] * mw_avg * -3.4125305230275943e-01 * rho_inv);
  jac[INDEX(90)] -= (h[INDEX(3)] * (working_temp * jac[INDEX(94)] - (j_temp * (cp[INDEX(8)] - cp[INDEX(9)]) * spec_rates[INDEX(3)] * 3.19988000e+01)));
  jac[INDEX(15)] += (spec_rates[INDEX(4)] * mw_avg * 7.8295700204607348e+00 * rho_inv);
  jac[INDEX(10)] -= (h[INDEX(4)] * (working_temp * jac[INDEX(15)] - (j_temp * (cp[INDEX(0)] - cp[INDEX(9)]) * spec_rates[INDEX(4)] * 1.70073400e+01)));
  jac[INDEX(25)] += (spec_rates[INDEX(4)] * mw_avg * 1.6266252759512660e+01 * rho_inv);
  jac[INDEX(20)] -= (h[INDEX(4)] * (working_temp * jac[INDEX(25)] - (j_temp * (cp[INDEX(1)] - cp[INDEX(9)]) * spec_rates[INDEX(4)] * 1.70073400e+01)));
  jac[INDEX(35)] += (spec_rates[INDEX(4)] * mw_avg * 4.5588589385677430e-01 * rho_inv);
  jac[INDEX(30)] -= (h[INDEX(4)] * (working_temp * jac[INDEX(35)] - (j_temp * (cp[INDEX(2)] - cp[INDEX(9)]) * spec_rates[INDEX(4)] * 1.70073400e+01)));
  jac[INDEX(45)] += (spec_rates[INDEX(4)] * mw_avg * -7.5613412367209046e-02 * rho_inv);
  jac[INDEX(40)] -= (h[INDEX(4)] * (working_temp * jac[INDEX(45)] - (j_temp * (cp[INDEX(3)] - cp[INDEX(9)]) * spec_rates[INDEX(4)] * 1.70073400e+01)));
  jac[INDEX(55)] += (spec_rates[INDEX(4)] * mw_avg * 3.9288728140880758e-01 * rho_inv);
  jac[INDEX(50)] -= (h[INDEX(4)] * (working_temp * jac[INDEX(55)] - (j_temp * (cp[INDEX(4)] - cp[INDEX(9)]) * spec_rates[INDEX(4)] * 1.70073400e+01)));
  jac[INDEX(65)] += (spec_rates[INDEX(4)] * mw_avg * 3.3693810937262486e-01 * rho_inv);
  jac[INDEX(60)] -= (h[INDEX(4)] * (working_temp * jac[INDEX(65)] - (j_temp * (cp[INDEX(5)] - cp[INDEX(9)]) * spec_rates[INDEX(4)] * 1.70073400e+01)));
  jac[INDEX(75)] += (spec_rates[INDEX(4)] * mw_avg * -9.1844018925609089e-02 * rho_inv);
  jac[INDEX(70)] -= (h[INDEX(4)] * (working_temp * jac[INDEX(75)] - (j_temp * (cp[INDEX(6)] - cp[INDEX(9)]) * spec_rates[INDEX(4)] * 1.70073400e+01)));
  jac[INDEX(85)] += (spec_rates[INDEX(4)] * mw_avg * -1.0711271859119242e-01 * rho_inv);
  jac[INDEX(80)] -= (h[INDEX(4)] * (working_temp * jac[INDEX(85)] - (j_temp * (cp[INDEX(7)] - cp[INDEX(9)]) * spec_rates[INDEX(4)] * 1.70073400e+01)));
  jac[INDEX(95)] += (spec_rates[INDEX(4)] * mw_avg * -1.8137576054573337e-01 * rho_inv);
  jac[INDEX(90)] -= (h[INDEX(4)] * (working_temp * jac[INDEX(95)] - (j_temp * (cp[INDEX(8)] - cp[INDEX(9)]) * spec_rates[INDEX(4)] * 1.70073400e+01)));
  jac[INDEX(16)] += (spec_rates[INDEX(5)] * mw_avg * 8.2935894853754846e+00 * rho_inv);
  jac[INDEX(10)] -= (h[INDEX(5)] * (working_temp * jac[INDEX(16)] - (j_temp * (cp[INDEX(0)] - cp[INDEX(9)]) * spec_rates[INDEX(5)] * 1.80152800e+01)));
  jac[INDEX(26)] += (spec_rates[INDEX(5)] * mw_avg * 1.7230272224427413e+01 * rho_inv);
  jac[INDEX(20)] -= (h[INDEX(5)] * (working_temp * jac[INDEX(26)] - (j_temp * (cp[INDEX(1)] - cp[INDEX(9)]) * spec_rates[INDEX(5)] * 1.80152800e+01)));
  jac[INDEX(36)] += (spec_rates[INDEX(5)] * mw_avg * 4.8290397121948930e-01 * rho_inv);
  jac[INDEX(30)] -= (h[INDEX(5)] * (working_temp * jac[INDEX(36)] - (j_temp * (cp[INDEX(2)] - cp[INDEX(9)]) * spec_rates[INDEX(5)] * 1.80152800e+01)));
  jac[INDEX(46)] += (spec_rates[INDEX(5)] * mw_avg * -8.0094641228477453e-02 * rho_inv);
  jac[INDEX(40)] -= (h[INDEX(5)] * (working_temp * jac[INDEX(46)] - (j_temp * (cp[INDEX(3)] - cp[INDEX(9)]) * spec_rates[INDEX(5)] * 1.80152800e+01)));
  jac[INDEX(56)] += (spec_rates[INDEX(5)] * mw_avg * 4.1617174602368529e-01 * rho_inv);
  jac[INDEX(50)] -= (h[INDEX(5)] * (working_temp * jac[INDEX(56)] - (j_temp * (cp[INDEX(4)] - cp[INDEX(9)]) * spec_rates[INDEX(5)] * 1.80152800e+01)));
  jac[INDEX(66)] += (spec_rates[INDEX(5)] * mw_avg * 3.5690674632355568e-01 * rho_inv);
  jac[INDEX(60)] -= (h[INDEX(5)] * (working_temp * jac[INDEX(66)] - (j_temp * (cp[INDEX(5)] - cp[INDEX(9)]) * spec_rates[INDEX(5)] * 1.80152800e+01)));
  jac[INDEX(76)] += (spec_rates[INDEX(5)] * mw_avg * -9.7287154679694010e-02 * rho_inv);
  jac[INDEX(70)] -= (h[INDEX(5)] * (working_temp * jac[INDEX(76)] - (j_temp * (cp[INDEX(6)] - cp[INDEX(9)]) * spec_rates[INDEX(5)] * 1.80152800e+01)));
  jac[INDEX(86)] += (spec_rates[INDEX(5)] * mw_avg * -1.1346075382637950e-01 * rho_inv);
  jac[INDEX(80)] -= (h[INDEX(5)] * (working_temp * jac[INDEX(86)] - (j_temp * (cp[INDEX(7)] - cp[INDEX(9)]) * spec_rates[INDEX(5)] * 1.80152800e+01)));
  jac[INDEX(96)] += (spec_rates[INDEX(5)] * mw_avg * -1.9212499494008703e-01 * rho_inv);
  jac[INDEX(90)] -= (h[INDEX(5)] * (working_temp * jac[INDEX(96)] - (j_temp * (cp[INDEX(8)] - cp[INDEX(9)]) * spec_rates[INDEX(5)] * 1.80152800e+01)));
  jac[INDEX(17)] += (spec_rates[INDEX(6)] * mw_avg * 1.5195120576006721e+01 * rho_inv);
  jac[INDEX(10)] -= (h[INDEX(6)] * (working_temp * jac[INDEX(17)] - (j_temp * (cp[INDEX(0)] - cp[INDEX(9)]) * spec_rates[INDEX(6)] * 3.30067400e+01)));
  jac[INDEX(27)] += (spec_rates[INDEX(6)] * mw_avg * 3.1568486054110576e+01 * rho_inv);
  jac[INDEX(20)] -= (h[INDEX(6)] * (working_temp * jac[INDEX(27)] - (j_temp * (cp[INDEX(1)] - cp[INDEX(9)]) * spec_rates[INDEX(6)] * 3.30067400e+01)));
  jac[INDEX(37)] += (spec_rates[INDEX(6)] * mw_avg * 8.8475371035083372e-01 * rho_inv);
  jac[INDEX(30)] -= (h[INDEX(6)] * (working_temp * jac[INDEX(37)] - (j_temp * (cp[INDEX(2)] - cp[INDEX(9)]) * spec_rates[INDEX(6)] * 3.30067400e+01)));
  jac[INDEX(47)] += (spec_rates[INDEX(6)] * mw_avg * -1.4674559587314970e-01 * rho_inv);
  jac[INDEX(40)] -= (h[INDEX(6)] * (working_temp * jac[INDEX(47)] - (j_temp * (cp[INDEX(3)] - cp[INDEX(9)]) * spec_rates[INDEX(6)] * 3.30067400e+01)));
  jac[INDEX(57)] += (spec_rates[INDEX(6)] * mw_avg * 7.6249009820273750e-01 * rho_inv);
  jac[INDEX(50)] -= (h[INDEX(6)] * (working_temp * jac[INDEX(57)] - (j_temp * (cp[INDEX(4)] - cp[INDEX(9)]) * spec_rates[INDEX(6)] * 3.30067400e+01)));
  jac[INDEX(67)] += (spec_rates[INDEX(6)] * mw_avg * 6.5390758179431885e-01 * rho_inv);
  jac[INDEX(60)] -= (h[INDEX(6)] * (working_temp * jac[INDEX(67)] - (j_temp * (cp[INDEX(5)] - cp[INDEX(9)]) * spec_rates[INDEX(6)] * 3.30067400e+01)));
  jac[INDEX(77)] += (spec_rates[INDEX(6)] * mw_avg * -1.7824490209713328e-01 * rho_inv);
  jac[INDEX(70)] -= (h[INDEX(6)] * (working_temp * jac[INDEX(77)] - (j_temp * (cp[INDEX(6)] - cp[INDEX(9)]) * spec_rates[INDEX(6)] * 3.30067400e+01)));
  jac[INDEX(87)] += (spec_rates[INDEX(6)] * mw_avg * -2.0787740194719778e-01 * rho_inv);
  jac[INDEX(80)] -= (h[INDEX(6)] * (working_temp * jac[INDEX(87)] - (j_temp * (cp[INDEX(7)] - cp[INDEX(9)]) * spec_rates[INDEX(6)] * 3.30067400e+01)));
  jac[INDEX(97)] += (spec_rates[INDEX(6)] * mw_avg * -3.5200228669711314e-01 * rho_inv);
  jac[INDEX(90)] -= (h[INDEX(6)] * (working_temp * jac[INDEX(97)] - (j_temp * (cp[INDEX(8)] - cp[INDEX(9)]) * spec_rates[INDEX(6)] * 3.30067400e+01)));
  jac[INDEX(18)] += (spec_rates[INDEX(7)] * mw_avg * 1.5659140040921470e+01 * rho_inv);
  jac[INDEX(10)] -= (h[INDEX(7)] * (working_temp * jac[INDEX(18)] - (j_temp * (cp[INDEX(0)] - cp[INDEX(9)]) * spec_rates[INDEX(7)] * 3.40146800e+01)));
  jac[INDEX(28)] += (spec_rates[INDEX(7)] * mw_avg * 3.2532505519025321e+01 * rho_inv);
  jac[INDEX(20)] -= (h[INDEX(7)] * (working_temp * jac[INDEX(28)] - (j_temp * (cp[INDEX(1)] - cp[INDEX(9)]) * spec_rates[INDEX(7)] * 3.40146800e+01)));
  jac[INDEX(38)] += (spec_rates[INDEX(7)] * mw_avg * 9.1177178771354861e-01 * rho_inv);
  jac[INDEX(30)] -= (h[INDEX(7)] * (working_temp * jac[INDEX(38)] - (j_temp * (cp[INDEX(2)] - cp[INDEX(9)]) * spec_rates[INDEX(7)] * 3.40146800e+01)));
  jac[INDEX(48)] += (spec_rates[INDEX(7)] * mw_avg * -1.5122682473441809e-01 * rho_inv);
  jac[INDEX(40)] -= (h[INDEX(7)] * (working_temp * jac[INDEX(48)] - (j_temp * (cp[INDEX(3)] - cp[INDEX(9)]) * spec_rates[INDEX(7)] * 3.40146800e+01)));
  jac[INDEX(58)] += (spec_rates[INDEX(7)] * mw_avg * 7.8577456281761515e-01 * rho_inv);
  jac[INDEX(50)] -= (h[INDEX(7)] * (working_temp * jac[INDEX(58)] - (j_temp * (cp[INDEX(4)] - cp[INDEX(9)]) * spec_rates[INDEX(7)] * 3.40146800e+01)));
  jac[INDEX(68)] += (spec_rates[INDEX(7)] * mw_avg * 6.7387621874524972e-01 * rho_inv);
  jac[INDEX(60)] -= (h[INDEX(7)] * (working_temp * jac[INDEX(68)] - (j_temp * (cp[INDEX(5)] - cp[INDEX(9)]) * spec_rates[INDEX(7)] * 3.40146800e+01)));
  jac[INDEX(78)] += (spec_rates[INDEX(7)] * mw_avg * -1.8368803785121818e-01 * rho_inv);
  jac[INDEX(70)] -= (h[INDEX(7)] * (working_temp * jac[INDEX(78)] - (j_temp * (cp[INDEX(6)] - cp[INDEX(9)]) * spec_rates[INDEX(7)] * 3.40146800e+01)));
  jac[INDEX(88)] += (spec_rates[INDEX(7)] * mw_avg * -2.1422543718238485e-01 * rho_inv);
  jac[INDEX(80)] -= (h[INDEX(7)] * (working_temp * jac[INDEX(88)] - (j_temp * (cp[INDEX(7)] - cp[INDEX(9)]) * spec_rates[INDEX(7)] * 3.40146800e+01)));
  jac[INDEX(98)] += (spec_rates[INDEX(7)] * mw_avg * -3.6275152109146674e-01 * rho_inv);
  jac[INDEX(90)] -= (h[INDEX(7)] * (working_temp * jac[INDEX(98)] - (j_temp * (cp[INDEX(8)] - cp[INDEX(9)]) * spec_rates[INDEX(7)] * 3.40146800e+01)));
  if (T <= 1000.0) {
    working_temp = (y[INDEX(1)] * 4.1244826576978785e+03 * (7.9805207499999992e-03 + T * (-3.8956301999999997e-05 + T * (6.0471628200000004e-08 + -2.9504470440000002e-11 * T))))
    + (y[INDEX(2)] * 8.2489653153957570e+03 * (7.0533281899999999e-13 + T * (-3.9918392799999999e-15 + T * (6.9024489599999992e-18 + -3.7109293280000002e-21 * T))))
    + (y[INDEX(3)] * 5.1967336900133762e+02 * (-3.2793188399999999e-03 + T * (1.3286127919999999e-05 + T * (-1.8384198720000000e-08 + 8.4506388399999997e-12 * T))))
    + (y[INDEX(4)] * 2.5983668450066881e+02 * (-2.9967341599999998e-03 + T * (1.9694604020000000e-05 + T * (-2.9043885270000002e-08 + 1.2974913480000000e-11 * T))))
    + (y[INDEX(5)] * 4.8887492694330808e+02 * (-2.4013175200000001e-03 + T * (9.2358768199999993e-06 + T * (-1.1643399990000001e-08 + 5.4564588000000001e-12 * T))))
    + (y[INDEX(6)] * 4.6152277955158070e+02 * (-2.0364341000000002e-03 + T * (1.3040804219999999e-05 + T * (-1.6463911860000002e-08 + 7.0879126800000000e-12 * T))))
    + (y[INDEX(7)] * 2.5190194790518544e+02 * (-4.7491205099999996e-03 + T * (4.2316578199999997e-05 + T * (-7.2829168200000001e-08 + 3.7169004960000001e-11 * T))))
    + (y[INDEX(8)] * 2.4443746347165404e+02 * (-5.4282241699999995e-04 + T * (3.3467140200000003e-05 + T * (-6.4731243900000002e-08 + 3.4498174519999999e-11 * T))))
    + (y[INDEX(9)] * 2.0813212426154001e+02 * (0.0000000000000000e+00 + T * (0.0000000000000000e+00 + T * (0.0000000000000000e+00 + 0.0000000000000000e+00 * T))))
    + (y_N * 2.9680218594762238e+02 * (1.4082404000000000e-03 + T * (-7.9264439999999996e-06 + T * (1.6924545000000002e-08 + -9.7794159999999997e-12 * T))));
  } else {
    working_temp = (y[INDEX(1)] * 4.1244826576978785e+03 * (-4.9402473099999998e-05 + T * (9.9891355599999992e-07 + T * (-5.3869918199999997e-10 + 8.0102150399999999e-14 * T))))
    + (y[INDEX(2)] * 8.2489653153957570e+03 * (-2.3084297299999999e-11 + T * (3.2312389600000002e-14 + T * (-1.4205457050000002e-17 + 1.9927894280000001e-21 * T))))
    + (y[INDEX(3)] * 5.1967336900133762e+02 * (-8.5974113700000004e-05 + T * (8.3896917800000004e-08 + T * (-3.0053339700000001e-11 + 4.9133476399999997e-15 * T))))
    + (y[INDEX(4)] * 2.5983668450066881e+02 * (1.4830875399999999e-03 + T * (-1.5159333380000000e-06 + T * (6.2841166499999998e-10 + -8.6687117599999996e-14 * T))))
    + (y[INDEX(5)] * 4.8887492694330808e+02 * (5.4842971599999998e-04 + T * (2.5301045599999997e-07 + T * (-2.6383846680000000e-10 + 4.6964950400000001e-14 * T))))
    + (y[INDEX(6)] * 4.6152277955158070e+02 * (2.1769180400000002e-03 + T * (-3.2814503599999998e-07 + T * (-2.9112596100000000e-10 + 6.7280396799999996e-14 * T))))
    + (y[INDEX(7)] * 2.5190194790518544e+02 * (2.2398201299999999e-03 + T * (-1.2673163000000000e-06 + T * (3.4273911000000001e-10 + -4.3163414000000000e-14 * T))))
    + (y[INDEX(8)] * 2.4443746347165404e+02 * (4.9083169400000000e-03 + T * (-3.8027844999999999e-06 + T * (1.1135579579999999e-09 + -1.1516332200000001e-13 * T))))
    + (y[INDEX(9)] * 2.0813212426154001e+02 * (0.0000000000000000e+00 + T * (0.0000000000000000e+00 + T * (0.0000000000000000e+00 + 0.0000000000000000e+00 * T))))
    + (y_N * 2.9680218594762238e+02 * (1.4879768000000001e-03 + T * (-1.1369520000000000e-06 + T * (3.0291114000000000e-10 + -2.7013403999999999e-14 * T))));
  }


  //Complete dT wrt T calculations
  jac[INDEX(0)] = -(spec_rates[INDEX(0)] * 2.01588000e+00 * (-working_temp * h[INDEX(0)] / cp_avg + cp[INDEX(0)]) + jac[INDEX(1)] * h[INDEX(0)] * rho
    + spec_rates[INDEX(1)] * 1.00794000e+00 * (-working_temp * h[INDEX(1)] / cp_avg + cp[INDEX(1)]) + jac[INDEX(2)] * h[INDEX(1)] * rho
    + spec_rates[INDEX(2)] * 1.59994000e+01 * (-working_temp * h[INDEX(2)] / cp_avg + cp[INDEX(2)]) + jac[INDEX(3)] * h[INDEX(2)] * rho
    + spec_rates[INDEX(3)] * 3.19988000e+01 * (-working_temp * h[INDEX(3)] / cp_avg + cp[INDEX(3)]) + jac[INDEX(4)] * h[INDEX(3)] * rho
    + spec_rates[INDEX(4)] * 1.70073400e+01 * (-working_temp * h[INDEX(4)] / cp_avg + cp[INDEX(4)]) + jac[INDEX(5)] * h[INDEX(4)] * rho
    + spec_rates[INDEX(5)] * 1.80152800e+01 * (-working_temp * h[INDEX(5)] / cp_avg + cp[INDEX(5)]) + jac[INDEX(6)] * h[INDEX(5)] * rho
    + spec_rates[INDEX(6)] * 3.30067400e+01 * (-working_temp * h[INDEX(6)] / cp_avg + cp[INDEX(6)]) + jac[INDEX(7)] * h[INDEX(6)] * rho
    + spec_rates[INDEX(7)] * 3.40146800e+01 * (-working_temp * h[INDEX(7)] / cp_avg + cp[INDEX(7)]) + jac[INDEX(8)] * h[INDEX(7)] * rho
    + spec_rates[INDEX(8)] * 3.99480000e+01 * (-working_temp * h[INDEX(8)] / cp_avg + cp[INDEX(8)]) + jac[INDEX(9)] * h[INDEX(8)] * rho
    + spec_rates[INDEX(9)] * 2.80134800e+01 * (-working_temp * h[INDEX(9)] / cp_avg + cp[INDEX(9)])) / (rho * cp_avg);
} // end eval_jacob

#elif defined(CONV)

__device__ void eval_jacob (const double t, const double rho, const double * __restrict__ y, double * __restrict__ jac, const mechanism_memory * __restrict__ d_mem) {

  extern volatile __shared__ double shared_temp[];
  double T = y[INDEX(0)];

  // average molecular weight
  double mw_avg;
  // pressure (derived from rho)
  double pres;
  // species molar concentrations
  double * __restrict__ conc = d_mem->conc;
  double y_N;
  eval_conc_rho(y[INDEX(0)], rho, &y[GRID_DIM], &y_N, &mw_avg, &pres, conc);

  // evaluate reaction rates
  double * __restrict__ fwd_rates = d_mem->fwd_rates;
  double * __restrict__ rev_rates = d_mem->rev_rates;
  eval_rxn_rates (T, pres, conc, fwd_rates, rev_rates);

  double * __restrict__ pres_mod = d_mem->pres_mod;
  //get pressure modifications to reaction rates
  get_rxn_pres_mod (T, pres, conc, pres_mod);

  // evaluate rate of change of species molar concentration
  double * __restrict__ spec_rates = d_mem->spec_rates;
  eval_spec_rates (fwd_rates, rev_rates, pres_mod, spec_rates, &spec_rates[INDEX(9)]);

  register double m = pres / (8.31446210e+03 * T);
  register double conc_temp;
  register double logT = log(T);
  register double j_temp = 0.0;
  register double kf = 0.0;
  register double pres_mod_temp = 0.0;
  register double Kc = 0.0;
  register double kr = 0;

  // NOTE: CONV analytical Jacobian implementation in progress
  // Infrastructure in place; full computation coming...
  // For now, zero the Jacobian as a placeholder
  memset(jac, 0, 10*10*sizeof(double));
} // end eval_jacob

#endif
