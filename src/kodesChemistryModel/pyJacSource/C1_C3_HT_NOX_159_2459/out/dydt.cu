#include "header.cuh"
#include "chem_utils.cuh"
#include "rates.cuh"
#include "gpu_memory.cuh"

#if defined(CONP)

__device__ void dydt (const double t, const double pres, const double * __restrict__ y, double * __restrict__ dy, const mechanism_memory * __restrict__ d_mem) {

  // species molar concentrations
  double * __restrict__ conc = d_mem->conc;
  double y_N;
  double mw_avg;
  double rho;
  eval_conc (y[INDEX(0)], pres, &y[GRID_DIM], &y_N, &mw_avg, &rho, conc);

  double * __restrict__ fwd_rates = d_mem->fwd_rates;
  double * __restrict__ rev_rates = d_mem->rev_rates;
  eval_rxn_rates (y[INDEX(0)], pres, conc, fwd_rates, rev_rates);

  // get pressure modifications to reaction rates
  double * __restrict__ pres_mod = d_mem->pres_mod;
  get_rxn_pres_mod (y[INDEX(0)], pres, conc, pres_mod);

  double * __restrict__ spec_rates = d_mem->spec_rates;
  // evaluate species molar net production rates
  eval_spec_rates (fwd_rates, rev_rates, pres_mod, spec_rates, &spec_rates[INDEX(158)]);
  // local array holding constant pressure specific heat
  double * __restrict__ cp = d_mem->cp;
  eval_cp (y[INDEX(0)], cp);

  // constant pressure mass-average specific heat
  double cp_avg = (cp[INDEX(0)] * y[INDEX(1)]) + (cp[INDEX(1)] * y[INDEX(2)])
              + (cp[INDEX(2)] * y[INDEX(3)]) + (cp[INDEX(3)] * y[INDEX(4)])
              + (cp[INDEX(4)] * y[INDEX(5)]) + (cp[INDEX(5)] * y[INDEX(6)])
              + (cp[INDEX(6)] * y[INDEX(7)]) + (cp[INDEX(7)] * y[INDEX(8)])
              + (cp[INDEX(8)] * y[INDEX(9)]) + (cp[INDEX(9)] * y[INDEX(10)])
              + (cp[INDEX(10)] * y[INDEX(11)]) + (cp[INDEX(11)] * y[INDEX(12)])
              + (cp[INDEX(12)] * y[INDEX(13)]) + (cp[INDEX(13)] * y[INDEX(14)])
              + (cp[INDEX(14)] * y[INDEX(15)]) + (cp[INDEX(15)] * y[INDEX(16)])
              + (cp[INDEX(16)] * y[INDEX(17)]) + (cp[INDEX(17)] * y[INDEX(18)])
              + (cp[INDEX(18)] * y[INDEX(19)]) + (cp[INDEX(19)] * y[INDEX(20)])
              + (cp[INDEX(20)] * y[INDEX(21)]) + (cp[INDEX(21)] * y[INDEX(22)])
              + (cp[INDEX(22)] * y[INDEX(23)]) + (cp[INDEX(23)] * y[INDEX(24)])
              + (cp[INDEX(24)] * y[INDEX(25)]) + (cp[INDEX(25)] * y[INDEX(26)])
              + (cp[INDEX(26)] * y[INDEX(27)]) + (cp[INDEX(27)] * y[INDEX(28)])
              + (cp[INDEX(28)] * y[INDEX(29)]) + (cp[INDEX(29)] * y[INDEX(30)])
              + (cp[INDEX(30)] * y[INDEX(31)]) + (cp[INDEX(31)] * y[INDEX(32)])
              + (cp[INDEX(32)] * y[INDEX(33)]) + (cp[INDEX(33)] * y[INDEX(34)])
              + (cp[INDEX(34)] * y[INDEX(35)]) + (cp[INDEX(35)] * y[INDEX(36)])
              + (cp[INDEX(36)] * y[INDEX(37)]) + (cp[INDEX(37)] * y[INDEX(38)])
              + (cp[INDEX(38)] * y[INDEX(39)]) + (cp[INDEX(39)] * y[INDEX(40)])
              + (cp[INDEX(40)] * y[INDEX(41)]) + (cp[INDEX(41)] * y[INDEX(42)])
              + (cp[INDEX(42)] * y[INDEX(43)]) + (cp[INDEX(43)] * y[INDEX(44)])
              + (cp[INDEX(44)] * y[INDEX(45)]) + (cp[INDEX(45)] * y[INDEX(46)])
              + (cp[INDEX(46)] * y[INDEX(47)]) + (cp[INDEX(47)] * y[INDEX(48)])
              + (cp[INDEX(48)] * y[INDEX(49)]) + (cp[INDEX(49)] * y[INDEX(50)])
              + (cp[INDEX(50)] * y[INDEX(51)]) + (cp[INDEX(51)] * y[INDEX(52)])
              + (cp[INDEX(52)] * y[INDEX(53)]) + (cp[INDEX(53)] * y[INDEX(54)])
              + (cp[INDEX(54)] * y[INDEX(55)]) + (cp[INDEX(55)] * y[INDEX(56)])
              + (cp[INDEX(56)] * y[INDEX(57)]) + (cp[INDEX(57)] * y[INDEX(58)])
              + (cp[INDEX(58)] * y[INDEX(59)]) + (cp[INDEX(59)] * y[INDEX(60)])
              + (cp[INDEX(60)] * y[INDEX(61)]) + (cp[INDEX(61)] * y[INDEX(62)])
              + (cp[INDEX(62)] * y[INDEX(63)]) + (cp[INDEX(63)] * y[INDEX(64)])
              + (cp[INDEX(64)] * y[INDEX(65)]) + (cp[INDEX(65)] * y[INDEX(66)])
              + (cp[INDEX(66)] * y[INDEX(67)]) + (cp[INDEX(67)] * y[INDEX(68)])
              + (cp[INDEX(68)] * y[INDEX(69)]) + (cp[INDEX(69)] * y[INDEX(70)])
              + (cp[INDEX(70)] * y[INDEX(71)]) + (cp[INDEX(71)] * y[INDEX(72)])
              + (cp[INDEX(72)] * y[INDEX(73)]) + (cp[INDEX(73)] * y[INDEX(74)])
              + (cp[INDEX(74)] * y[INDEX(75)]) + (cp[INDEX(75)] * y[INDEX(76)])
              + (cp[INDEX(76)] * y[INDEX(77)]) + (cp[INDEX(77)] * y[INDEX(78)])
              + (cp[INDEX(78)] * y[INDEX(79)]) + (cp[INDEX(79)] * y[INDEX(80)])
              + (cp[INDEX(80)] * y[INDEX(81)]) + (cp[INDEX(81)] * y[INDEX(82)])
              + (cp[INDEX(82)] * y[INDEX(83)]) + (cp[INDEX(83)] * y[INDEX(84)])
              + (cp[INDEX(84)] * y[INDEX(85)]) + (cp[INDEX(85)] * y[INDEX(86)])
              + (cp[INDEX(86)] * y[INDEX(87)]) + (cp[INDEX(87)] * y[INDEX(88)])
              + (cp[INDEX(88)] * y[INDEX(89)]) + (cp[INDEX(89)] * y[INDEX(90)])
              + (cp[INDEX(90)] * y[INDEX(91)]) + (cp[INDEX(91)] * y[INDEX(92)])
              + (cp[INDEX(92)] * y[INDEX(93)]) + (cp[INDEX(93)] * y[INDEX(94)])
              + (cp[INDEX(94)] * y[INDEX(95)]) + (cp[INDEX(95)] * y[INDEX(96)])
              + (cp[INDEX(96)] * y[INDEX(97)]) + (cp[INDEX(97)] * y[INDEX(98)])
              + (cp[INDEX(98)] * y[INDEX(99)]) + (cp[INDEX(99)] * y[INDEX(100)])
              + (cp[INDEX(100)] * y[INDEX(101)]) + (cp[INDEX(101)] * y[INDEX(102)])
              + (cp[INDEX(102)] * y[INDEX(103)]) + (cp[INDEX(103)] * y[INDEX(104)])
              + (cp[INDEX(104)] * y[INDEX(105)]) + (cp[INDEX(105)] * y[INDEX(106)])
              + (cp[INDEX(106)] * y[INDEX(107)]) + (cp[INDEX(107)] * y[INDEX(108)])
              + (cp[INDEX(108)] * y[INDEX(109)]) + (cp[INDEX(109)] * y[INDEX(110)])
              + (cp[INDEX(110)] * y[INDEX(111)]) + (cp[INDEX(111)] * y[INDEX(112)])
              + (cp[INDEX(112)] * y[INDEX(113)]) + (cp[INDEX(113)] * y[INDEX(114)])
              + (cp[INDEX(114)] * y[INDEX(115)]) + (cp[INDEX(115)] * y[INDEX(116)])
              + (cp[INDEX(116)] * y[INDEX(117)]) + (cp[INDEX(117)] * y[INDEX(118)])
              + (cp[INDEX(118)] * y[INDEX(119)]) + (cp[INDEX(119)] * y[INDEX(120)])
              + (cp[INDEX(120)] * y[INDEX(121)]) + (cp[INDEX(121)] * y[INDEX(122)])
              + (cp[INDEX(122)] * y[INDEX(123)]) + (cp[INDEX(123)] * y[INDEX(124)])
              + (cp[INDEX(124)] * y[INDEX(125)]) + (cp[INDEX(125)] * y[INDEX(126)])
              + (cp[INDEX(126)] * y[INDEX(127)]) + (cp[INDEX(127)] * y[INDEX(128)])
              + (cp[INDEX(128)] * y[INDEX(129)]) + (cp[INDEX(129)] * y[INDEX(130)])
              + (cp[INDEX(130)] * y[INDEX(131)]) + (cp[INDEX(131)] * y[INDEX(132)])
              + (cp[INDEX(132)] * y[INDEX(133)]) + (cp[INDEX(133)] * y[INDEX(134)])
              + (cp[INDEX(134)] * y[INDEX(135)]) + (cp[INDEX(135)] * y[INDEX(136)])
              + (cp[INDEX(136)] * y[INDEX(137)]) + (cp[INDEX(137)] * y[INDEX(138)])
              + (cp[INDEX(138)] * y[INDEX(139)]) + (cp[INDEX(139)] * y[INDEX(140)])
              + (cp[INDEX(140)] * y[INDEX(141)]) + (cp[INDEX(141)] * y[INDEX(142)])
              + (cp[INDEX(142)] * y[INDEX(143)]) + (cp[INDEX(143)] * y[INDEX(144)])
              + (cp[INDEX(144)] * y[INDEX(145)]) + (cp[INDEX(145)] * y[INDEX(146)])
              + (cp[INDEX(146)] * y[INDEX(147)]) + (cp[INDEX(147)] * y[INDEX(148)])
              + (cp[INDEX(148)] * y[INDEX(149)]) + (cp[INDEX(149)] * y[INDEX(150)])
              + (cp[INDEX(150)] * y[INDEX(151)]) + (cp[INDEX(151)] * y[INDEX(152)])
              + (cp[INDEX(152)] * y[INDEX(153)]) + (cp[INDEX(153)] * y[INDEX(154)])
              + (cp[INDEX(154)] * y[INDEX(155)]) + (cp[INDEX(155)] * y[INDEX(156)])
              + (cp[INDEX(156)] * y[INDEX(157)]) + (cp[INDEX(157)] * y[INDEX(158)]) + (cp[INDEX(158)] * y_N);

  // local array for species enthalpies
  double * __restrict__ h = d_mem->h;
  eval_h(y[INDEX(0)], h);
  // rate of change of temperature
  dy[INDEX(0)] = (-1.0 / (rho * cp_avg)) * ((spec_rates[INDEX(2)] * h[INDEX(2)] * 2.0158800000000001e+00)
        + (spec_rates[INDEX(3)] * h[INDEX(3)] * 1.0079400000000001e+00)
        + (spec_rates[INDEX(4)] * h[INDEX(4)] * 3.1998799999999999e+01)
        + (spec_rates[INDEX(5)] * h[INDEX(5)] * 1.5999400000000000e+01)
        + (spec_rates[INDEX(6)] * h[INDEX(6)] * 1.8015280000000001e+01)
        + (spec_rates[INDEX(7)] * h[INDEX(7)] * 1.7007339999999999e+01)
        + (spec_rates[INDEX(8)] * h[INDEX(8)] * 3.4014679999999998e+01)
        + (spec_rates[INDEX(9)] * h[INDEX(9)] * 3.3006740000000001e+01)
        + (spec_rates[INDEX(10)] * h[INDEX(10)] * 2.8010399999999997e+01)
        + (spec_rates[INDEX(11)] * h[INDEX(11)] * 4.4009799999999998e+01)
        + (spec_rates[INDEX(12)] * h[INDEX(12)] * 4.5017739999999996e+01)
        + (spec_rates[INDEX(13)] * h[INDEX(13)] * 1.6042760000000001e+01)
        + (spec_rates[INDEX(14)] * h[INDEX(14)] * 1.5034820000000000e+01)
        + (spec_rates[INDEX(15)] * h[INDEX(15)] * 1.4026879999999998e+01)
        + (spec_rates[INDEX(16)] * h[INDEX(16)] * 1.4026879999999998e+01)
        + (spec_rates[INDEX(17)] * h[INDEX(17)] * 1.2010999999999999e+01)
        + (spec_rates[INDEX(18)] * h[INDEX(18)] * 1.3018939999999999e+01)
        + (spec_rates[INDEX(19)] * h[INDEX(19)] * 4.8041560000000004e+01)
        + (spec_rates[INDEX(20)] * h[INDEX(20)] * 4.7033619999999999e+01)
        + (spec_rates[INDEX(21)] * h[INDEX(21)] * 3.2042160000000003e+01)
        + (spec_rates[INDEX(22)] * h[INDEX(22)] * 3.1034219999999998e+01)
        + (spec_rates[INDEX(23)] * h[INDEX(23)] * 3.1034219999999998e+01)
        + (spec_rates[INDEX(24)] * h[INDEX(24)] * 3.0026280000000000e+01)
        + (spec_rates[INDEX(25)] * h[INDEX(25)] * 2.9018339999999998e+01)
        + (spec_rates[INDEX(26)] * h[INDEX(26)] * 6.2025079999999996e+01)
        + (spec_rates[INDEX(27)] * h[INDEX(27)] * 4.6025679999999994e+01)
        + (spec_rates[INDEX(28)] * h[INDEX(28)] * 4.5017739999999996e+01)
        + (spec_rates[INDEX(29)] * h[INDEX(29)] * 3.0069640000000000e+01)
        + (spec_rates[INDEX(30)] * h[INDEX(30)] * 2.9061699999999998e+01)
        + (spec_rates[INDEX(31)] * h[INDEX(31)] * 6.2068439999999995e+01)
        + (spec_rates[INDEX(32)] * h[INDEX(32)] * 6.1060499999999998e+01)
        + (spec_rates[INDEX(33)] * h[INDEX(33)] * 2.8053759999999997e+01)
        + (spec_rates[INDEX(34)] * h[INDEX(34)] * 2.7045819999999999e+01)
        + (spec_rates[INDEX(35)] * h[INDEX(35)] * 2.6037879999999998e+01)
        + (spec_rates[INDEX(36)] * h[INDEX(36)] * 2.5029940000000000e+01)
        + (spec_rates[INDEX(37)] * h[INDEX(37)] * 4.6069040000000001e+01)
        + (spec_rates[INDEX(38)] * h[INDEX(38)] * 4.5061099999999996e+01)
        + (spec_rates[INDEX(39)] * h[INDEX(39)] * 4.5061099999999996e+01)
        + (spec_rates[INDEX(40)] * h[INDEX(40)] * 4.5061099999999996e+01)
        + (spec_rates[INDEX(41)] * h[INDEX(41)] * 6.1060499999999998e+01)
        + (spec_rates[INDEX(42)] * h[INDEX(42)] * 4.4053159999999998e+01)
        + (spec_rates[INDEX(43)] * h[INDEX(43)] * 4.3045220000000000e+01)
        + (spec_rates[INDEX(44)] * h[INDEX(44)] * 4.4053159999999998e+01)
        + (spec_rates[INDEX(45)] * h[INDEX(45)] * 4.3045220000000000e+01)
        + (spec_rates[INDEX(46)] * h[INDEX(46)] * 4.3045220000000000e+01)
        + (spec_rates[INDEX(47)] * h[INDEX(47)] * 4.2037279999999996e+01)
        + (spec_rates[INDEX(48)] * h[INDEX(48)] * 4.1029339999999998e+01)
        + (spec_rates[INDEX(49)] * h[INDEX(49)] * 4.6025679999999994e+01)
        + (spec_rates[INDEX(50)] * h[INDEX(50)] * 7.5044019999999989e+01)
        + (spec_rates[INDEX(52)] * h[INDEX(52)] * 6.0052560000000000e+01)
        + (spec_rates[INDEX(53)] * h[INDEX(53)] * 5.8036679999999997e+01)
        + (spec_rates[INDEX(54)] * h[INDEX(54)] * 9.3059299999999993e+01)
        + (spec_rates[INDEX(55)] * h[INDEX(55)] * 7.6051959999999994e+01)
        + (spec_rates[INDEX(56)] * h[INDEX(56)] * 6.0052560000000000e+01)
        + (spec_rates[INDEX(57)] * h[INDEX(57)] * 5.9044619999999995e+01)
        + (spec_rates[INDEX(58)] * h[INDEX(58)] * 4.4096519999999998e+01)
        + (spec_rates[INDEX(59)] * h[INDEX(59)] * 4.3088580000000000e+01)
        + (spec_rates[INDEX(60)] * h[INDEX(60)] * 4.3088580000000000e+01)
        + (spec_rates[INDEX(61)] * h[INDEX(61)] * 4.2080640000000002e+01)
        + (spec_rates[INDEX(62)] * h[INDEX(62)] * 4.1072699999999998e+01)
        + (spec_rates[INDEX(63)] * h[INDEX(63)] * 4.1072699999999998e+01)
        + (spec_rates[INDEX(64)] * h[INDEX(64)] * 4.1072699999999998e+01)
        + (spec_rates[INDEX(65)] * h[INDEX(65)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(66)] * h[INDEX(66)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(67)] * h[INDEX(67)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(68)] * h[INDEX(68)] * 8.8106319999999997e+01)
        + (spec_rates[INDEX(69)] * h[INDEX(69)] * 5.6064160000000001e+01)
        + (spec_rates[INDEX(70)] * h[INDEX(70)] * 7.4079440000000005e+01)
        + (spec_rates[INDEX(71)] * h[INDEX(71)] * 5.9087980000000002e+01)
        + (spec_rates[INDEX(72)] * h[INDEX(72)] * 5.9087980000000002e+01)
        + (spec_rates[INDEX(73)] * h[INDEX(73)] * 9.1086780000000005e+01)
        + (spec_rates[INDEX(74)] * h[INDEX(74)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(75)] * h[INDEX(75)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(76)] * h[INDEX(76)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(77)] * h[INDEX(77)] * 4.0064760000000000e+01)
        + (spec_rates[INDEX(78)] * h[INDEX(78)] * 4.0064760000000000e+01)
        + (spec_rates[INDEX(79)] * h[INDEX(79)] * 3.9056820000000002e+01)
        + (spec_rates[INDEX(80)] * h[INDEX(80)] * 3.8048880000000004e+01)
        + (spec_rates[INDEX(81)] * h[INDEX(81)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(82)] * h[INDEX(82)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(83)] * h[INDEX(83)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(84)] * h[INDEX(84)] * 5.6064160000000001e+01)
        + (spec_rates[INDEX(85)] * h[INDEX(85)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(86)] * h[INDEX(86)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(87)] * h[INDEX(87)] * 5.8123399999999997e+01)
        + (spec_rates[INDEX(88)] * h[INDEX(88)] * 5.7115459999999999e+01)
        + (spec_rates[INDEX(89)] * h[INDEX(89)] * 5.7115459999999999e+01)
        + (spec_rates[INDEX(90)] * h[INDEX(90)] * 5.8123399999999997e+01)
        + (spec_rates[INDEX(91)] * h[INDEX(91)] * 5.7115459999999999e+01)
        + (spec_rates[INDEX(92)] * h[INDEX(92)] * 5.7115459999999999e+01)
        + (spec_rates[INDEX(93)] * h[INDEX(93)] * 5.6107519999999994e+01)
        + (spec_rates[INDEX(94)] * h[INDEX(94)] * 5.5099579999999996e+01)
        + (spec_rates[INDEX(95)] * h[INDEX(95)] * 7.1098979999999997e+01)
        + (spec_rates[INDEX(96)] * h[INDEX(96)] * 5.6107519999999994e+01)
        + (spec_rates[INDEX(97)] * h[INDEX(97)] * 5.6107519999999994e+01)
        + (spec_rates[INDEX(98)] * h[INDEX(98)] * 5.5099579999999996e+01)
        + (spec_rates[INDEX(99)] * h[INDEX(99)] * 5.5099579999999996e+01)
        + (spec_rates[INDEX(100)] * h[INDEX(100)] * 7.1098979999999997e+01)
        + (spec_rates[INDEX(101)] * h[INDEX(101)] * 5.4091639999999998e+01)
        + (spec_rates[INDEX(102)] * h[INDEX(102)] * 5.3083699999999993e+01)
        + (spec_rates[INDEX(103)] * h[INDEX(103)] * 5.2075759999999995e+01)
        + (spec_rates[INDEX(104)] * h[INDEX(104)] * 5.1067819999999998e+01)
        + (spec_rates[INDEX(105)] * h[INDEX(105)] * 5.0059880000000000e+01)
        + (spec_rates[INDEX(106)] * h[INDEX(106)] * 7.8113640000000004e+01)
        + (spec_rates[INDEX(107)] * h[INDEX(107)] * 7.8113640000000004e+01)
        + (spec_rates[INDEX(108)] * h[INDEX(108)] * 7.7105699999999999e+01)
        + (spec_rates[INDEX(109)] * h[INDEX(109)] * 6.6102639999999994e+01)
        + (spec_rates[INDEX(110)] * h[INDEX(110)] * 6.5094699999999989e+01)
        + (spec_rates[INDEX(111)] * h[INDEX(111)] * 8.0129519999999999e+01)
        + (spec_rates[INDEX(112)] * h[INDEX(112)] * 1.2817352000000000e+02)
        + (spec_rates[INDEX(113)] * h[INDEX(113)] * 3.0006140000000002e+01)
        + (spec_rates[INDEX(114)] * h[INDEX(114)] * 4.4012880000000003e+01)
        + (spec_rates[INDEX(115)] * h[INDEX(115)] * 4.6005539999999996e+01)
        + (spec_rates[INDEX(116)] * h[INDEX(116)] * 3.1014080000000000e+01)
        + (spec_rates[INDEX(117)] * h[INDEX(117)] * 4.7013480000000001e+01)
        + (spec_rates[INDEX(118)] * h[INDEX(118)] * 4.7013480000000001e+01)
        + (spec_rates[INDEX(119)] * h[INDEX(119)] * 6.3012879999999996e+01)
        + (spec_rates[INDEX(120)] * h[INDEX(120)] * 3.0029360000000000e+01)
        + (spec_rates[INDEX(121)] * h[INDEX(121)] * 3.0029360000000000e+01)
        + (spec_rates[INDEX(122)] * h[INDEX(122)] * 4.5020820000000001e+01)
        + (spec_rates[INDEX(123)] * h[INDEX(123)] * 4.6028759999999998e+01)
        + (spec_rates[INDEX(124)] * h[INDEX(124)] * 3.3029960000000003e+01)
        + (spec_rates[INDEX(125)] * h[INDEX(125)] * 3.2022019999999998e+01)
        + (spec_rates[INDEX(126)] * h[INDEX(126)] * 1.7030560000000001e+01)
        + (spec_rates[INDEX(127)] * h[INDEX(127)] * 3.2045240000000000e+01)
        + (spec_rates[INDEX(128)] * h[INDEX(128)] * 1.4006740000000001e+01)
        + (spec_rates[INDEX(129)] * h[INDEX(129)] * 6.2004939999999998e+01)
        + (spec_rates[INDEX(130)] * h[INDEX(130)] * 1.5014680000000000e+01)
        + (spec_rates[INDEX(131)] * h[INDEX(131)] * 2.9021420000000003e+01)
        + (spec_rates[INDEX(132)] * h[INDEX(132)] * 1.6022620000000000e+01)
        + (spec_rates[INDEX(133)] * h[INDEX(133)] * 3.2022019999999998e+01)
        + (spec_rates[INDEX(134)] * h[INDEX(134)] * 3.1037300000000002e+01)
        + (spec_rates[INDEX(135)] * h[INDEX(135)] * 2.7025680000000001e+01)
        + (spec_rates[INDEX(136)] * h[INDEX(136)] * 2.7025680000000001e+01)
        + (spec_rates[INDEX(137)] * h[INDEX(137)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(138)] * h[INDEX(138)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(139)] * h[INDEX(139)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(140)] * h[INDEX(140)] * 4.4033020000000000e+01)
        + (spec_rates[INDEX(141)] * h[INDEX(141)] * 4.5040959999999998e+01)
        + (spec_rates[INDEX(142)] * h[INDEX(142)] * 6.1040360000000000e+01)
        + (spec_rates[INDEX(143)] * h[INDEX(143)] * 6.1040360000000000e+01)
        + (spec_rates[INDEX(144)] * h[INDEX(144)] * 7.7039760000000001e+01)
        + (spec_rates[INDEX(145)] * h[INDEX(145)] * 4.1052560000000000e+01)
        + (spec_rates[INDEX(146)] * h[INDEX(146)] * 2.6017740000000000e+01)
        + (spec_rates[INDEX(147)] * h[INDEX(147)] * 4.0024479999999997e+01)
        + (spec_rates[INDEX(148)] * h[INDEX(148)] * 4.2017139999999998e+01)
        + (spec_rates[INDEX(149)] * h[INDEX(149)] * 4.1032420000000002e+01)
        + (spec_rates[INDEX(150)] * h[INDEX(150)] * 2.8033619999999999e+01)
        + (spec_rates[INDEX(151)] * h[INDEX(151)] * 2.8033619999999999e+01)
        + (spec_rates[INDEX(152)] * h[INDEX(152)] * 5.2035480000000000e+01)
        + (spec_rates[INDEX(153)] * h[INDEX(153)] * 4.0044619999999995e+01)
        + (spec_rates[INDEX(154)] * h[INDEX(154)] * 2.9041560000000000e+01)
        + (spec_rates[INDEX(155)] * h[INDEX(155)] * 3.1057440000000000e+01)
        + (spec_rates[INDEX(156)] * h[INDEX(156)] * 3.0049500000000002e+01)
        + (spec_rates[INDEX(157)] * h[INDEX(157)] * 3.0049500000000002e+01)
        + (spec_rates[INDEX(158)] * h[INDEX(158)] * 2.8013480000000001e+01));

  // calculate rate of change of species mass fractions
  dy[INDEX(1)] = spec_rates[INDEX(0)] * (3.9948000000000000e+01 / rho);
  dy[INDEX(2)] = spec_rates[INDEX(1)] * (4.0026000000000002e+00 / rho);
  dy[INDEX(3)] = spec_rates[INDEX(2)] * (2.0158800000000001e+00 / rho);
  dy[INDEX(4)] = spec_rates[INDEX(3)] * (1.0079400000000001e+00 / rho);
  dy[INDEX(5)] = spec_rates[INDEX(4)] * (3.1998799999999999e+01 / rho);
  dy[INDEX(6)] = spec_rates[INDEX(5)] * (1.5999400000000000e+01 / rho);
  dy[INDEX(7)] = spec_rates[INDEX(6)] * (1.8015280000000001e+01 / rho);
  dy[INDEX(8)] = spec_rates[INDEX(7)] * (1.7007339999999999e+01 / rho);
  dy[INDEX(9)] = spec_rates[INDEX(8)] * (3.4014679999999998e+01 / rho);
  dy[INDEX(10)] = spec_rates[INDEX(9)] * (3.3006740000000001e+01 / rho);
  dy[INDEX(11)] = spec_rates[INDEX(10)] * (2.8010399999999997e+01 / rho);
  dy[INDEX(12)] = spec_rates[INDEX(11)] * (4.4009799999999998e+01 / rho);
  dy[INDEX(13)] = spec_rates[INDEX(12)] * (4.5017739999999996e+01 / rho);
  dy[INDEX(14)] = spec_rates[INDEX(13)] * (1.6042760000000001e+01 / rho);
  dy[INDEX(15)] = spec_rates[INDEX(14)] * (1.5034820000000000e+01 / rho);
  dy[INDEX(16)] = spec_rates[INDEX(15)] * (1.4026879999999998e+01 / rho);
  dy[INDEX(17)] = spec_rates[INDEX(16)] * (1.4026879999999998e+01 / rho);
  dy[INDEX(18)] = spec_rates[INDEX(17)] * (1.2010999999999999e+01 / rho);
  dy[INDEX(19)] = spec_rates[INDEX(18)] * (1.3018939999999999e+01 / rho);
  dy[INDEX(20)] = spec_rates[INDEX(19)] * (4.8041560000000004e+01 / rho);
  dy[INDEX(21)] = spec_rates[INDEX(20)] * (4.7033619999999999e+01 / rho);
  dy[INDEX(22)] = spec_rates[INDEX(21)] * (3.2042160000000003e+01 / rho);
  dy[INDEX(23)] = spec_rates[INDEX(22)] * (3.1034219999999998e+01 / rho);
  dy[INDEX(24)] = spec_rates[INDEX(23)] * (3.1034219999999998e+01 / rho);
  dy[INDEX(25)] = spec_rates[INDEX(24)] * (3.0026280000000000e+01 / rho);
  dy[INDEX(26)] = spec_rates[INDEX(25)] * (2.9018339999999998e+01 / rho);
  dy[INDEX(27)] = spec_rates[INDEX(26)] * (6.2025079999999996e+01 / rho);
  dy[INDEX(28)] = spec_rates[INDEX(27)] * (4.6025679999999994e+01 / rho);
  dy[INDEX(29)] = spec_rates[INDEX(28)] * (4.5017739999999996e+01 / rho);
  dy[INDEX(30)] = spec_rates[INDEX(29)] * (3.0069640000000000e+01 / rho);
  dy[INDEX(31)] = spec_rates[INDEX(30)] * (2.9061699999999998e+01 / rho);
  dy[INDEX(32)] = spec_rates[INDEX(31)] * (6.2068439999999995e+01 / rho);
  dy[INDEX(33)] = spec_rates[INDEX(32)] * (6.1060499999999998e+01 / rho);
  dy[INDEX(34)] = spec_rates[INDEX(33)] * (2.8053759999999997e+01 / rho);
  dy[INDEX(35)] = spec_rates[INDEX(34)] * (2.7045819999999999e+01 / rho);
  dy[INDEX(36)] = spec_rates[INDEX(35)] * (2.6037879999999998e+01 / rho);
  dy[INDEX(37)] = spec_rates[INDEX(36)] * (2.5029940000000000e+01 / rho);
  dy[INDEX(38)] = spec_rates[INDEX(37)] * (4.6069040000000001e+01 / rho);
  dy[INDEX(39)] = spec_rates[INDEX(38)] * (4.5061099999999996e+01 / rho);
  dy[INDEX(40)] = spec_rates[INDEX(39)] * (4.5061099999999996e+01 / rho);
  dy[INDEX(41)] = spec_rates[INDEX(40)] * (4.5061099999999996e+01 / rho);
  dy[INDEX(42)] = spec_rates[INDEX(41)] * (6.1060499999999998e+01 / rho);
  dy[INDEX(43)] = spec_rates[INDEX(42)] * (4.4053159999999998e+01 / rho);
  dy[INDEX(44)] = spec_rates[INDEX(43)] * (4.3045220000000000e+01 / rho);
  dy[INDEX(45)] = spec_rates[INDEX(44)] * (4.4053159999999998e+01 / rho);
  dy[INDEX(46)] = spec_rates[INDEX(45)] * (4.3045220000000000e+01 / rho);
  dy[INDEX(47)] = spec_rates[INDEX(46)] * (4.3045220000000000e+01 / rho);
  dy[INDEX(48)] = spec_rates[INDEX(47)] * (4.2037279999999996e+01 / rho);
  dy[INDEX(49)] = spec_rates[INDEX(48)] * (4.1029339999999998e+01 / rho);
  dy[INDEX(50)] = spec_rates[INDEX(49)] * (4.6025679999999994e+01 / rho);
  dy[INDEX(51)] = spec_rates[INDEX(50)] * (7.5044019999999989e+01 / rho);
  dy[INDEX(52)] = spec_rates[INDEX(51)] * (7.6051959999999994e+01 / rho);
  dy[INDEX(53)] = spec_rates[INDEX(52)] * (6.0052560000000000e+01 / rho);
  dy[INDEX(54)] = spec_rates[INDEX(53)] * (5.8036679999999997e+01 / rho);
  dy[INDEX(55)] = spec_rates[INDEX(54)] * (9.3059299999999993e+01 / rho);
  dy[INDEX(56)] = spec_rates[INDEX(55)] * (7.6051959999999994e+01 / rho);
  dy[INDEX(57)] = spec_rates[INDEX(56)] * (6.0052560000000000e+01 / rho);
  dy[INDEX(58)] = spec_rates[INDEX(57)] * (5.9044619999999995e+01 / rho);
  dy[INDEX(59)] = spec_rates[INDEX(58)] * (4.4096519999999998e+01 / rho);
  dy[INDEX(60)] = spec_rates[INDEX(59)] * (4.3088580000000000e+01 / rho);
  dy[INDEX(61)] = spec_rates[INDEX(60)] * (4.3088580000000000e+01 / rho);
  dy[INDEX(62)] = spec_rates[INDEX(61)] * (4.2080640000000002e+01 / rho);
  dy[INDEX(63)] = spec_rates[INDEX(62)] * (4.1072699999999998e+01 / rho);
  dy[INDEX(64)] = spec_rates[INDEX(63)] * (4.1072699999999998e+01 / rho);
  dy[INDEX(65)] = spec_rates[INDEX(64)] * (4.1072699999999998e+01 / rho);
  dy[INDEX(66)] = spec_rates[INDEX(65)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(67)] = spec_rates[INDEX(66)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(68)] = spec_rates[INDEX(67)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(69)] = spec_rates[INDEX(68)] * (8.8106319999999997e+01 / rho);
  dy[INDEX(70)] = spec_rates[INDEX(69)] * (5.6064160000000001e+01 / rho);
  dy[INDEX(71)] = spec_rates[INDEX(70)] * (7.4079440000000005e+01 / rho);
  dy[INDEX(72)] = spec_rates[INDEX(71)] * (5.9087980000000002e+01 / rho);
  dy[INDEX(73)] = spec_rates[INDEX(72)] * (5.9087980000000002e+01 / rho);
  dy[INDEX(74)] = spec_rates[INDEX(73)] * (9.1086780000000005e+01 / rho);
  dy[INDEX(75)] = spec_rates[INDEX(74)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(76)] = spec_rates[INDEX(75)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(77)] = spec_rates[INDEX(76)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(78)] = spec_rates[INDEX(77)] * (4.0064760000000000e+01 / rho);
  dy[INDEX(79)] = spec_rates[INDEX(78)] * (4.0064760000000000e+01 / rho);
  dy[INDEX(80)] = spec_rates[INDEX(79)] * (3.9056820000000002e+01 / rho);
  dy[INDEX(81)] = spec_rates[INDEX(80)] * (3.8048880000000004e+01 / rho);
  dy[INDEX(82)] = spec_rates[INDEX(81)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(83)] = spec_rates[INDEX(82)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(84)] = spec_rates[INDEX(83)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(85)] = spec_rates[INDEX(84)] * (5.6064160000000001e+01 / rho);
  dy[INDEX(86)] = spec_rates[INDEX(85)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(87)] = spec_rates[INDEX(86)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(88)] = spec_rates[INDEX(87)] * (5.8123399999999997e+01 / rho);
  dy[INDEX(89)] = spec_rates[INDEX(88)] * (5.7115459999999999e+01 / rho);
  dy[INDEX(90)] = spec_rates[INDEX(89)] * (5.7115459999999999e+01 / rho);
  dy[INDEX(91)] = spec_rates[INDEX(90)] * (5.8123399999999997e+01 / rho);
  dy[INDEX(92)] = spec_rates[INDEX(91)] * (5.7115459999999999e+01 / rho);
  dy[INDEX(93)] = spec_rates[INDEX(92)] * (5.7115459999999999e+01 / rho);
  dy[INDEX(94)] = spec_rates[INDEX(93)] * (5.6107519999999994e+01 / rho);
  dy[INDEX(95)] = spec_rates[INDEX(94)] * (5.5099579999999996e+01 / rho);
  dy[INDEX(96)] = spec_rates[INDEX(95)] * (7.1098979999999997e+01 / rho);
  dy[INDEX(97)] = spec_rates[INDEX(96)] * (5.6107519999999994e+01 / rho);
  dy[INDEX(98)] = spec_rates[INDEX(97)] * (5.6107519999999994e+01 / rho);
  dy[INDEX(99)] = spec_rates[INDEX(98)] * (5.5099579999999996e+01 / rho);
  dy[INDEX(100)] = spec_rates[INDEX(99)] * (5.5099579999999996e+01 / rho);
  dy[INDEX(101)] = spec_rates[INDEX(100)] * (7.1098979999999997e+01 / rho);
  dy[INDEX(102)] = spec_rates[INDEX(101)] * (5.4091639999999998e+01 / rho);
  dy[INDEX(103)] = spec_rates[INDEX(102)] * (5.3083699999999993e+01 / rho);
  dy[INDEX(104)] = spec_rates[INDEX(103)] * (5.2075759999999995e+01 / rho);
  dy[INDEX(105)] = spec_rates[INDEX(104)] * (5.1067819999999998e+01 / rho);
  dy[INDEX(106)] = spec_rates[INDEX(105)] * (5.0059880000000000e+01 / rho);
  dy[INDEX(107)] = spec_rates[INDEX(106)] * (7.8113640000000004e+01 / rho);
  dy[INDEX(108)] = spec_rates[INDEX(107)] * (7.8113640000000004e+01 / rho);
  dy[INDEX(109)] = spec_rates[INDEX(108)] * (7.7105699999999999e+01 / rho);
  dy[INDEX(110)] = spec_rates[INDEX(109)] * (6.6102639999999994e+01 / rho);
  dy[INDEX(111)] = spec_rates[INDEX(110)] * (6.5094699999999989e+01 / rho);
  dy[INDEX(112)] = spec_rates[INDEX(111)] * (8.0129519999999999e+01 / rho);
  dy[INDEX(113)] = spec_rates[INDEX(112)] * (1.2817352000000000e+02 / rho);
  dy[INDEX(114)] = spec_rates[INDEX(113)] * (3.0006140000000002e+01 / rho);
  dy[INDEX(115)] = spec_rates[INDEX(114)] * (4.4012880000000003e+01 / rho);
  dy[INDEX(116)] = spec_rates[INDEX(115)] * (4.6005539999999996e+01 / rho);
  dy[INDEX(117)] = spec_rates[INDEX(116)] * (3.1014080000000000e+01 / rho);
  dy[INDEX(118)] = spec_rates[INDEX(117)] * (4.7013480000000001e+01 / rho);
  dy[INDEX(119)] = spec_rates[INDEX(118)] * (4.7013480000000001e+01 / rho);
  dy[INDEX(120)] = spec_rates[INDEX(119)] * (6.3012879999999996e+01 / rho);
  dy[INDEX(121)] = spec_rates[INDEX(120)] * (3.0029360000000000e+01 / rho);
  dy[INDEX(122)] = spec_rates[INDEX(121)] * (3.0029360000000000e+01 / rho);
  dy[INDEX(123)] = spec_rates[INDEX(122)] * (4.5020820000000001e+01 / rho);
  dy[INDEX(124)] = spec_rates[INDEX(123)] * (4.6028759999999998e+01 / rho);
  dy[INDEX(125)] = spec_rates[INDEX(124)] * (3.3029960000000003e+01 / rho);
  dy[INDEX(126)] = spec_rates[INDEX(125)] * (3.2022019999999998e+01 / rho);
  dy[INDEX(127)] = spec_rates[INDEX(126)] * (1.7030560000000001e+01 / rho);
  dy[INDEX(128)] = spec_rates[INDEX(127)] * (3.2045240000000000e+01 / rho);
  dy[INDEX(129)] = spec_rates[INDEX(128)] * (1.4006740000000001e+01 / rho);
  dy[INDEX(130)] = spec_rates[INDEX(129)] * (6.2004939999999998e+01 / rho);
  dy[INDEX(131)] = spec_rates[INDEX(130)] * (1.5014680000000000e+01 / rho);
  dy[INDEX(132)] = spec_rates[INDEX(131)] * (2.9021420000000003e+01 / rho);
  dy[INDEX(133)] = spec_rates[INDEX(132)] * (1.6022620000000000e+01 / rho);
  dy[INDEX(134)] = spec_rates[INDEX(133)] * (3.2022019999999998e+01 / rho);
  dy[INDEX(135)] = spec_rates[INDEX(134)] * (3.1037300000000002e+01 / rho);
  dy[INDEX(136)] = spec_rates[INDEX(135)] * (2.7025680000000001e+01 / rho);
  dy[INDEX(137)] = spec_rates[INDEX(136)] * (2.7025680000000001e+01 / rho);
  dy[INDEX(138)] = spec_rates[INDEX(137)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(139)] = spec_rates[INDEX(138)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(140)] = spec_rates[INDEX(139)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(141)] = spec_rates[INDEX(140)] * (4.4033020000000000e+01 / rho);
  dy[INDEX(142)] = spec_rates[INDEX(141)] * (4.5040959999999998e+01 / rho);
  dy[INDEX(143)] = spec_rates[INDEX(142)] * (6.1040360000000000e+01 / rho);
  dy[INDEX(144)] = spec_rates[INDEX(143)] * (6.1040360000000000e+01 / rho);
  dy[INDEX(145)] = spec_rates[INDEX(144)] * (7.7039760000000001e+01 / rho);
  dy[INDEX(146)] = spec_rates[INDEX(145)] * (4.1052560000000000e+01 / rho);
  dy[INDEX(147)] = spec_rates[INDEX(146)] * (2.6017740000000000e+01 / rho);
  dy[INDEX(148)] = spec_rates[INDEX(147)] * (4.0024479999999997e+01 / rho);
  dy[INDEX(149)] = spec_rates[INDEX(148)] * (4.2017139999999998e+01 / rho);
  dy[INDEX(150)] = spec_rates[INDEX(149)] * (4.1032420000000002e+01 / rho);
  dy[INDEX(151)] = spec_rates[INDEX(150)] * (2.8033619999999999e+01 / rho);
  dy[INDEX(152)] = spec_rates[INDEX(151)] * (2.8033619999999999e+01 / rho);
  dy[INDEX(153)] = spec_rates[INDEX(152)] * (5.2035480000000000e+01 / rho);
  dy[INDEX(154)] = spec_rates[INDEX(153)] * (4.0044619999999995e+01 / rho);
  dy[INDEX(155)] = spec_rates[INDEX(154)] * (2.9041560000000000e+01 / rho);
  dy[INDEX(156)] = spec_rates[INDEX(155)] * (3.1057440000000000e+01 / rho);
  dy[INDEX(157)] = spec_rates[INDEX(156)] * (3.0049500000000002e+01 / rho);
  dy[INDEX(158)] = spec_rates[INDEX(157)] * (3.0049500000000002e+01 / rho);

} // end dydt

#elif defined(CONV)

__device__ void dydt (const double t, const double rho, const double * __restrict__ y, double * __restrict__ dy, mechanism_memory * __restrict__ d_mem) {

  // species molar concentrations
  double * __restrict__ conc = d_mem->conc;
  double y_N;
  double mw_avg;
  double pres;
  eval_conc_rho (y[INDEX(0)]rho, &y[GRID_DIM], &y_N, &mw_avg, &pres, conc);

  double * __restrict__ fwd_rates = d_mem->fwd_rates;
  double * __restrict__ rev_rates = d_mem->rev_rates;
  eval_rxn_rates (y[INDEX(0)], pres, conc, fwd_rates, rev_rates);

  // get pressure modifications to reaction rates
  double * __restrict__ pres_mod = d_mem->pres_mod;
  get_rxn_pres_mod (y[INDEX(0)], pres, conc, pres_mod);

  // evaluate species molar net production rates
  double dy_N;  eval_spec_rates (fwd_rates, rev_rates, pres_mod, &dy[GRID_DIM], &dy_N);

  double * __restrict__ cv = d_mem->cp;
  eval_cv(y[INDEX(0)], cv);

  // constant volume mass-average specific heat
  double cv_avg = (cv[INDEX(0)] * y[INDEX(1)]) + (cv[INDEX(1)] * y[INDEX(2)])
              + (cv[INDEX(2)] * y[INDEX(3)]) + (cv[INDEX(3)] * y[INDEX(4)])
              + (cv[INDEX(4)] * y[INDEX(5)]) + (cv[INDEX(5)] * y[INDEX(6)])
              + (cv[INDEX(6)] * y[INDEX(7)]) + (cv[INDEX(7)] * y[INDEX(8)])
              + (cv[INDEX(8)] * y[INDEX(9)]) + (cv[INDEX(9)] * y[INDEX(10)])
              + (cv[INDEX(10)] * y[INDEX(11)]) + (cv[INDEX(11)] * y[INDEX(12)])
              + (cv[INDEX(12)] * y[INDEX(13)]) + (cv[INDEX(13)] * y[INDEX(14)])
              + (cv[INDEX(14)] * y[INDEX(15)]) + (cv[INDEX(15)] * y[INDEX(16)])
              + (cv[INDEX(16)] * y[INDEX(17)]) + (cv[INDEX(17)] * y[INDEX(18)])
              + (cv[INDEX(18)] * y[INDEX(19)]) + (cv[INDEX(19)] * y[INDEX(20)])
              + (cv[INDEX(20)] * y[INDEX(21)]) + (cv[INDEX(21)] * y[INDEX(22)])
              + (cv[INDEX(22)] * y[INDEX(23)]) + (cv[INDEX(23)] * y[INDEX(24)])
              + (cv[INDEX(24)] * y[INDEX(25)]) + (cv[INDEX(25)] * y[INDEX(26)])
              + (cv[INDEX(26)] * y[INDEX(27)]) + (cv[INDEX(27)] * y[INDEX(28)])
              + (cv[INDEX(28)] * y[INDEX(29)]) + (cv[INDEX(29)] * y[INDEX(30)])
              + (cv[INDEX(30)] * y[INDEX(31)]) + (cv[INDEX(31)] * y[INDEX(32)])
              + (cv[INDEX(32)] * y[INDEX(33)]) + (cv[INDEX(33)] * y[INDEX(34)])
              + (cv[INDEX(34)] * y[INDEX(35)]) + (cv[INDEX(35)] * y[INDEX(36)])
              + (cv[INDEX(36)] * y[INDEX(37)]) + (cv[INDEX(37)] * y[INDEX(38)])
              + (cv[INDEX(38)] * y[INDEX(39)]) + (cv[INDEX(39)] * y[INDEX(40)])
              + (cv[INDEX(40)] * y[INDEX(41)]) + (cv[INDEX(41)] * y[INDEX(42)])
              + (cv[INDEX(42)] * y[INDEX(43)]) + (cv[INDEX(43)] * y[INDEX(44)])
              + (cv[INDEX(44)] * y[INDEX(45)]) + (cv[INDEX(45)] * y[INDEX(46)])
              + (cv[INDEX(46)] * y[INDEX(47)]) + (cv[INDEX(47)] * y[INDEX(48)])
              + (cv[INDEX(48)] * y[INDEX(49)]) + (cv[INDEX(49)] * y[INDEX(50)])
              + (cv[INDEX(50)] * y[INDEX(51)]) + (cv[INDEX(51)] * y[INDEX(52)])
              + (cv[INDEX(52)] * y[INDEX(53)]) + (cv[INDEX(53)] * y[INDEX(54)])
              + (cv[INDEX(54)] * y[INDEX(55)]) + (cv[INDEX(55)] * y[INDEX(56)])
              + (cv[INDEX(56)] * y[INDEX(57)]) + (cv[INDEX(57)] * y[INDEX(58)])
              + (cv[INDEX(58)] * y[INDEX(59)]) + (cv[INDEX(59)] * y[INDEX(60)])
              + (cv[INDEX(60)] * y[INDEX(61)]) + (cv[INDEX(61)] * y[INDEX(62)])
              + (cv[INDEX(62)] * y[INDEX(63)]) + (cv[INDEX(63)] * y[INDEX(64)])
              + (cv[INDEX(64)] * y[INDEX(65)]) + (cv[INDEX(65)] * y[INDEX(66)])
              + (cv[INDEX(66)] * y[INDEX(67)]) + (cv[INDEX(67)] * y[INDEX(68)])
              + (cv[INDEX(68)] * y[INDEX(69)]) + (cv[INDEX(69)] * y[INDEX(70)])
              + (cv[INDEX(70)] * y[INDEX(71)]) + (cv[INDEX(71)] * y[INDEX(72)])
              + (cv[INDEX(72)] * y[INDEX(73)]) + (cv[INDEX(73)] * y[INDEX(74)])
              + (cv[INDEX(74)] * y[INDEX(75)]) + (cv[INDEX(75)] * y[INDEX(76)])
              + (cv[INDEX(76)] * y[INDEX(77)]) + (cv[INDEX(77)] * y[INDEX(78)])
              + (cv[INDEX(78)] * y[INDEX(79)]) + (cv[INDEX(79)] * y[INDEX(80)])
              + (cv[INDEX(80)] * y[INDEX(81)]) + (cv[INDEX(81)] * y[INDEX(82)])
              + (cv[INDEX(82)] * y[INDEX(83)]) + (cv[INDEX(83)] * y[INDEX(84)])
              + (cv[INDEX(84)] * y[INDEX(85)]) + (cv[INDEX(85)] * y[INDEX(86)])
              + (cv[INDEX(86)] * y[INDEX(87)]) + (cv[INDEX(87)] * y[INDEX(88)])
              + (cv[INDEX(88)] * y[INDEX(89)]) + (cv[INDEX(89)] * y[INDEX(90)])
              + (cv[INDEX(90)] * y[INDEX(91)]) + (cv[INDEX(91)] * y[INDEX(92)])
              + (cv[INDEX(92)] * y[INDEX(93)]) + (cv[INDEX(93)] * y[INDEX(94)])
              + (cv[INDEX(94)] * y[INDEX(95)]) + (cv[INDEX(95)] * y[INDEX(96)])
              + (cv[INDEX(96)] * y[INDEX(97)]) + (cv[INDEX(97)] * y[INDEX(98)])
              + (cv[INDEX(98)] * y[INDEX(99)]) + (cv[INDEX(99)] * y[INDEX(100)])
              + (cv[INDEX(100)] * y[INDEX(101)]) + (cv[INDEX(101)] * y[INDEX(102)])
              + (cv[INDEX(102)] * y[INDEX(103)]) + (cv[INDEX(103)] * y[INDEX(104)])
              + (cv[INDEX(104)] * y[INDEX(105)]) + (cv[INDEX(105)] * y[INDEX(106)])
              + (cv[INDEX(106)] * y[INDEX(107)]) + (cv[INDEX(107)] * y[INDEX(108)])
              + (cv[INDEX(108)] * y[INDEX(109)]) + (cv[INDEX(109)] * y[INDEX(110)])
              + (cv[INDEX(110)] * y[INDEX(111)]) + (cv[INDEX(111)] * y[INDEX(112)])
              + (cv[INDEX(112)] * y[INDEX(113)]) + (cv[INDEX(113)] * y[INDEX(114)])
              + (cv[INDEX(114)] * y[INDEX(115)]) + (cv[INDEX(115)] * y[INDEX(116)])
              + (cv[INDEX(116)] * y[INDEX(117)]) + (cv[INDEX(117)] * y[INDEX(118)])
              + (cv[INDEX(118)] * y[INDEX(119)]) + (cv[INDEX(119)] * y[INDEX(120)])
              + (cv[INDEX(120)] * y[INDEX(121)]) + (cv[INDEX(121)] * y[INDEX(122)])
              + (cv[INDEX(122)] * y[INDEX(123)]) + (cv[INDEX(123)] * y[INDEX(124)])
              + (cv[INDEX(124)] * y[INDEX(125)]) + (cv[INDEX(125)] * y[INDEX(126)])
              + (cv[INDEX(126)] * y[INDEX(127)]) + (cv[INDEX(127)] * y[INDEX(128)])
              + (cv[INDEX(128)] * y[INDEX(129)]) + (cv[INDEX(129)] * y[INDEX(130)])
              + (cv[INDEX(130)] * y[INDEX(131)]) + (cv[INDEX(131)] * y[INDEX(132)])
              + (cv[INDEX(132)] * y[INDEX(133)]) + (cv[INDEX(133)] * y[INDEX(134)])
              + (cv[INDEX(134)] * y[INDEX(135)]) + (cv[INDEX(135)] * y[INDEX(136)])
              + (cv[INDEX(136)] * y[INDEX(137)]) + (cv[INDEX(137)] * y[INDEX(138)])
              + (cv[INDEX(138)] * y[INDEX(139)]) + (cv[INDEX(139)] * y[INDEX(140)])
              + (cv[INDEX(140)] * y[INDEX(141)]) + (cv[INDEX(141)] * y[INDEX(142)])
              + (cv[INDEX(142)] * y[INDEX(143)]) + (cv[INDEX(143)] * y[INDEX(144)])
              + (cv[INDEX(144)] * y[INDEX(145)]) + (cv[INDEX(145)] * y[INDEX(146)])
              + (cv[INDEX(146)] * y[INDEX(147)]) + (cv[INDEX(147)] * y[INDEX(148)])
              + (cv[INDEX(148)] * y[INDEX(149)]) + (cv[INDEX(149)] * y[INDEX(150)])
              + (cv[INDEX(150)] * y[INDEX(151)]) + (cv[INDEX(151)] * y[INDEX(152)])
              + (cv[INDEX(152)] * y[INDEX(153)]) + (cv[INDEX(153)] * y[INDEX(154)])
              + (cv[INDEX(154)] * y[INDEX(155)]) + (cv[INDEX(155)] * y[INDEX(156)])
              + (cv[INDEX(156)] * y[INDEX(157)]) + (cv[INDEX(157)] * y[INDEX(158)])(cv[INDEX(158)] * y_N);

  // local array for species internal energies
  double * __restrict__ u = d_mem->h;
  eval_u (y[INDEX(0)], u);

  // rate of change of temperature
  dy[INDEX(0)] = (-1.0 / (rho * cv_avg)) * ((spec_rates[INDEX(2)] * u[INDEX(2)] * 2.0158800000000001e+00)
        + (spec_rates[INDEX(3)] * u[INDEX(3)] * 1.0079400000000001e+00)
        + (spec_rates[INDEX(4)] * u[INDEX(4)] * 3.1998799999999999e+01)
        + (spec_rates[INDEX(5)] * u[INDEX(5)] * 1.5999400000000000e+01)
        + (spec_rates[INDEX(6)] * u[INDEX(6)] * 1.8015280000000001e+01)
        + (spec_rates[INDEX(7)] * u[INDEX(7)] * 1.7007339999999999e+01)
        + (spec_rates[INDEX(8)] * u[INDEX(8)] * 3.4014679999999998e+01)
        + (spec_rates[INDEX(9)] * u[INDEX(9)] * 3.3006740000000001e+01)
        + (spec_rates[INDEX(10)] * u[INDEX(10)] * 2.8010399999999997e+01)
        + (spec_rates[INDEX(11)] * u[INDEX(11)] * 4.4009799999999998e+01)
        + (spec_rates[INDEX(12)] * u[INDEX(12)] * 4.5017739999999996e+01)
        + (spec_rates[INDEX(13)] * u[INDEX(13)] * 1.6042760000000001e+01)
        + (spec_rates[INDEX(14)] * u[INDEX(14)] * 1.5034820000000000e+01)
        + (spec_rates[INDEX(15)] * u[INDEX(15)] * 1.4026879999999998e+01)
        + (spec_rates[INDEX(16)] * u[INDEX(16)] * 1.4026879999999998e+01)
        + (spec_rates[INDEX(17)] * u[INDEX(17)] * 1.2010999999999999e+01)
        + (spec_rates[INDEX(18)] * u[INDEX(18)] * 1.3018939999999999e+01)
        + (spec_rates[INDEX(19)] * u[INDEX(19)] * 4.8041560000000004e+01)
        + (spec_rates[INDEX(20)] * u[INDEX(20)] * 4.7033619999999999e+01)
        + (spec_rates[INDEX(21)] * u[INDEX(21)] * 3.2042160000000003e+01)
        + (spec_rates[INDEX(22)] * u[INDEX(22)] * 3.1034219999999998e+01)
        + (spec_rates[INDEX(23)] * u[INDEX(23)] * 3.1034219999999998e+01)
        + (spec_rates[INDEX(24)] * u[INDEX(24)] * 3.0026280000000000e+01)
        + (spec_rates[INDEX(25)] * u[INDEX(25)] * 2.9018339999999998e+01)
        + (spec_rates[INDEX(26)] * u[INDEX(26)] * 6.2025079999999996e+01)
        + (spec_rates[INDEX(27)] * u[INDEX(27)] * 4.6025679999999994e+01)
        + (spec_rates[INDEX(28)] * u[INDEX(28)] * 4.5017739999999996e+01)
        + (spec_rates[INDEX(29)] * u[INDEX(29)] * 3.0069640000000000e+01)
        + (spec_rates[INDEX(30)] * u[INDEX(30)] * 2.9061699999999998e+01)
        + (spec_rates[INDEX(31)] * u[INDEX(31)] * 6.2068439999999995e+01)
        + (spec_rates[INDEX(32)] * u[INDEX(32)] * 6.1060499999999998e+01)
        + (spec_rates[INDEX(33)] * u[INDEX(33)] * 2.8053759999999997e+01)
        + (spec_rates[INDEX(34)] * u[INDEX(34)] * 2.7045819999999999e+01)
        + (spec_rates[INDEX(35)] * u[INDEX(35)] * 2.6037879999999998e+01)
        + (spec_rates[INDEX(36)] * u[INDEX(36)] * 2.5029940000000000e+01)
        + (spec_rates[INDEX(37)] * u[INDEX(37)] * 4.6069040000000001e+01)
        + (spec_rates[INDEX(38)] * u[INDEX(38)] * 4.5061099999999996e+01)
        + (spec_rates[INDEX(39)] * u[INDEX(39)] * 4.5061099999999996e+01)
        + (spec_rates[INDEX(40)] * u[INDEX(40)] * 4.5061099999999996e+01)
        + (spec_rates[INDEX(41)] * u[INDEX(41)] * 6.1060499999999998e+01)
        + (spec_rates[INDEX(42)] * u[INDEX(42)] * 4.4053159999999998e+01)
        + (spec_rates[INDEX(43)] * u[INDEX(43)] * 4.3045220000000000e+01)
        + (spec_rates[INDEX(44)] * u[INDEX(44)] * 4.4053159999999998e+01)
        + (spec_rates[INDEX(45)] * u[INDEX(45)] * 4.3045220000000000e+01)
        + (spec_rates[INDEX(46)] * u[INDEX(46)] * 4.3045220000000000e+01)
        + (spec_rates[INDEX(47)] * u[INDEX(47)] * 4.2037279999999996e+01)
        + (spec_rates[INDEX(48)] * u[INDEX(48)] * 4.1029339999999998e+01)
        + (spec_rates[INDEX(49)] * u[INDEX(49)] * 4.6025679999999994e+01)
        + (spec_rates[INDEX(50)] * u[INDEX(50)] * 7.5044019999999989e+01)
        + (spec_rates[INDEX(52)] * u[INDEX(52)] * 6.0052560000000000e+01)
        + (spec_rates[INDEX(53)] * u[INDEX(53)] * 5.8036679999999997e+01)
        + (spec_rates[INDEX(54)] * u[INDEX(54)] * 9.3059299999999993e+01)
        + (spec_rates[INDEX(55)] * u[INDEX(55)] * 7.6051959999999994e+01)
        + (spec_rates[INDEX(56)] * u[INDEX(56)] * 6.0052560000000000e+01)
        + (spec_rates[INDEX(57)] * u[INDEX(57)] * 5.9044619999999995e+01)
        + (spec_rates[INDEX(58)] * u[INDEX(58)] * 4.4096519999999998e+01)
        + (spec_rates[INDEX(59)] * u[INDEX(59)] * 4.3088580000000000e+01)
        + (spec_rates[INDEX(60)] * u[INDEX(60)] * 4.3088580000000000e+01)
        + (spec_rates[INDEX(61)] * u[INDEX(61)] * 4.2080640000000002e+01)
        + (spec_rates[INDEX(62)] * u[INDEX(62)] * 4.1072699999999998e+01)
        + (spec_rates[INDEX(63)] * u[INDEX(63)] * 4.1072699999999998e+01)
        + (spec_rates[INDEX(64)] * u[INDEX(64)] * 4.1072699999999998e+01)
        + (spec_rates[INDEX(65)] * u[INDEX(65)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(66)] * u[INDEX(66)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(67)] * u[INDEX(67)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(68)] * u[INDEX(68)] * 8.8106319999999997e+01)
        + (spec_rates[INDEX(69)] * u[INDEX(69)] * 5.6064160000000001e+01)
        + (spec_rates[INDEX(70)] * u[INDEX(70)] * 7.4079440000000005e+01)
        + (spec_rates[INDEX(71)] * u[INDEX(71)] * 5.9087980000000002e+01)
        + (spec_rates[INDEX(72)] * u[INDEX(72)] * 5.9087980000000002e+01)
        + (spec_rates[INDEX(73)] * u[INDEX(73)] * 9.1086780000000005e+01)
        + (spec_rates[INDEX(74)] * u[INDEX(74)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(75)] * u[INDEX(75)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(76)] * u[INDEX(76)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(77)] * u[INDEX(77)] * 4.0064760000000000e+01)
        + (spec_rates[INDEX(78)] * u[INDEX(78)] * 4.0064760000000000e+01)
        + (spec_rates[INDEX(79)] * u[INDEX(79)] * 3.9056820000000002e+01)
        + (spec_rates[INDEX(80)] * u[INDEX(80)] * 3.8048880000000004e+01)
        + (spec_rates[INDEX(81)] * u[INDEX(81)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(82)] * u[INDEX(82)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(83)] * u[INDEX(83)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(84)] * u[INDEX(84)] * 5.6064160000000001e+01)
        + (spec_rates[INDEX(85)] * u[INDEX(85)] * 5.8080040000000004e+01)
        + (spec_rates[INDEX(86)] * u[INDEX(86)] * 5.7072099999999999e+01)
        + (spec_rates[INDEX(87)] * u[INDEX(87)] * 5.8123399999999997e+01)
        + (spec_rates[INDEX(88)] * u[INDEX(88)] * 5.7115459999999999e+01)
        + (spec_rates[INDEX(89)] * u[INDEX(89)] * 5.7115459999999999e+01)
        + (spec_rates[INDEX(90)] * u[INDEX(90)] * 5.8123399999999997e+01)
        + (spec_rates[INDEX(91)] * u[INDEX(91)] * 5.7115459999999999e+01)
        + (spec_rates[INDEX(92)] * u[INDEX(92)] * 5.7115459999999999e+01)
        + (spec_rates[INDEX(93)] * u[INDEX(93)] * 5.6107519999999994e+01)
        + (spec_rates[INDEX(94)] * u[INDEX(94)] * 5.5099579999999996e+01)
        + (spec_rates[INDEX(95)] * u[INDEX(95)] * 7.1098979999999997e+01)
        + (spec_rates[INDEX(96)] * u[INDEX(96)] * 5.6107519999999994e+01)
        + (spec_rates[INDEX(97)] * u[INDEX(97)] * 5.6107519999999994e+01)
        + (spec_rates[INDEX(98)] * u[INDEX(98)] * 5.5099579999999996e+01)
        + (spec_rates[INDEX(99)] * u[INDEX(99)] * 5.5099579999999996e+01)
        + (spec_rates[INDEX(100)] * u[INDEX(100)] * 7.1098979999999997e+01)
        + (spec_rates[INDEX(101)] * u[INDEX(101)] * 5.4091639999999998e+01)
        + (spec_rates[INDEX(102)] * u[INDEX(102)] * 5.3083699999999993e+01)
        + (spec_rates[INDEX(103)] * u[INDEX(103)] * 5.2075759999999995e+01)
        + (spec_rates[INDEX(104)] * u[INDEX(104)] * 5.1067819999999998e+01)
        + (spec_rates[INDEX(105)] * u[INDEX(105)] * 5.0059880000000000e+01)
        + (spec_rates[INDEX(106)] * u[INDEX(106)] * 7.8113640000000004e+01)
        + (spec_rates[INDEX(107)] * u[INDEX(107)] * 7.8113640000000004e+01)
        + (spec_rates[INDEX(108)] * u[INDEX(108)] * 7.7105699999999999e+01)
        + (spec_rates[INDEX(109)] * u[INDEX(109)] * 6.6102639999999994e+01)
        + (spec_rates[INDEX(110)] * u[INDEX(110)] * 6.5094699999999989e+01)
        + (spec_rates[INDEX(111)] * u[INDEX(111)] * 8.0129519999999999e+01)
        + (spec_rates[INDEX(112)] * u[INDEX(112)] * 1.2817352000000000e+02)
        + (spec_rates[INDEX(113)] * u[INDEX(113)] * 3.0006140000000002e+01)
        + (spec_rates[INDEX(114)] * u[INDEX(114)] * 4.4012880000000003e+01)
        + (spec_rates[INDEX(115)] * u[INDEX(115)] * 4.6005539999999996e+01)
        + (spec_rates[INDEX(116)] * u[INDEX(116)] * 3.1014080000000000e+01)
        + (spec_rates[INDEX(117)] * u[INDEX(117)] * 4.7013480000000001e+01)
        + (spec_rates[INDEX(118)] * u[INDEX(118)] * 4.7013480000000001e+01)
        + (spec_rates[INDEX(119)] * u[INDEX(119)] * 6.3012879999999996e+01)
        + (spec_rates[INDEX(120)] * u[INDEX(120)] * 3.0029360000000000e+01)
        + (spec_rates[INDEX(121)] * u[INDEX(121)] * 3.0029360000000000e+01)
        + (spec_rates[INDEX(122)] * u[INDEX(122)] * 4.5020820000000001e+01)
        + (spec_rates[INDEX(123)] * u[INDEX(123)] * 4.6028759999999998e+01)
        + (spec_rates[INDEX(124)] * u[INDEX(124)] * 3.3029960000000003e+01)
        + (spec_rates[INDEX(125)] * u[INDEX(125)] * 3.2022019999999998e+01)
        + (spec_rates[INDEX(126)] * u[INDEX(126)] * 1.7030560000000001e+01)
        + (spec_rates[INDEX(127)] * u[INDEX(127)] * 3.2045240000000000e+01)
        + (spec_rates[INDEX(128)] * u[INDEX(128)] * 1.4006740000000001e+01)
        + (spec_rates[INDEX(129)] * u[INDEX(129)] * 6.2004939999999998e+01)
        + (spec_rates[INDEX(130)] * u[INDEX(130)] * 1.5014680000000000e+01)
        + (spec_rates[INDEX(131)] * u[INDEX(131)] * 2.9021420000000003e+01)
        + (spec_rates[INDEX(132)] * u[INDEX(132)] * 1.6022620000000000e+01)
        + (spec_rates[INDEX(133)] * u[INDEX(133)] * 3.2022019999999998e+01)
        + (spec_rates[INDEX(134)] * u[INDEX(134)] * 3.1037300000000002e+01)
        + (spec_rates[INDEX(135)] * u[INDEX(135)] * 2.7025680000000001e+01)
        + (spec_rates[INDEX(136)] * u[INDEX(136)] * 2.7025680000000001e+01)
        + (spec_rates[INDEX(137)] * u[INDEX(137)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(138)] * u[INDEX(138)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(139)] * u[INDEX(139)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(140)] * u[INDEX(140)] * 4.4033020000000000e+01)
        + (spec_rates[INDEX(141)] * u[INDEX(141)] * 4.5040959999999998e+01)
        + (spec_rates[INDEX(142)] * u[INDEX(142)] * 6.1040360000000000e+01)
        + (spec_rates[INDEX(143)] * u[INDEX(143)] * 6.1040360000000000e+01)
        + (spec_rates[INDEX(144)] * u[INDEX(144)] * 7.7039760000000001e+01)
        + (spec_rates[INDEX(145)] * u[INDEX(145)] * 4.1052560000000000e+01)
        + (spec_rates[INDEX(146)] * u[INDEX(146)] * 2.6017740000000000e+01)
        + (spec_rates[INDEX(147)] * u[INDEX(147)] * 4.0024479999999997e+01)
        + (spec_rates[INDEX(148)] * u[INDEX(148)] * 4.2017139999999998e+01)
        + (spec_rates[INDEX(149)] * u[INDEX(149)] * 4.1032420000000002e+01)
        + (spec_rates[INDEX(150)] * u[INDEX(150)] * 2.8033619999999999e+01)
        + (spec_rates[INDEX(151)] * u[INDEX(151)] * 2.8033619999999999e+01)
        + (spec_rates[INDEX(152)] * u[INDEX(152)] * 5.2035480000000000e+01)
        + (spec_rates[INDEX(153)] * u[INDEX(153)] * 4.0044619999999995e+01)
        + (spec_rates[INDEX(154)] * u[INDEX(154)] * 2.9041560000000000e+01)
        + (spec_rates[INDEX(155)] * u[INDEX(155)] * 3.1057440000000000e+01)
        + (spec_rates[INDEX(156)] * u[INDEX(156)] * 3.0049500000000002e+01)
        + (spec_rates[INDEX(157)] * u[INDEX(157)] * 3.0049500000000002e+01)
        + (spec_rates[INDEX(158)] * u[INDEX(158)] * 2.8013480000000001e+01));

  // calculate rate of change of species mass fractions
  dy[INDEX(1)] = spec_rates[INDEX(0)] * (3.9948000000000000e+01 / rho);
  dy[INDEX(2)] = spec_rates[INDEX(1)] * (4.0026000000000002e+00 / rho);
  dy[INDEX(3)] = spec_rates[INDEX(2)] * (2.0158800000000001e+00 / rho);
  dy[INDEX(4)] = spec_rates[INDEX(3)] * (1.0079400000000001e+00 / rho);
  dy[INDEX(5)] = spec_rates[INDEX(4)] * (3.1998799999999999e+01 / rho);
  dy[INDEX(6)] = spec_rates[INDEX(5)] * (1.5999400000000000e+01 / rho);
  dy[INDEX(7)] = spec_rates[INDEX(6)] * (1.8015280000000001e+01 / rho);
  dy[INDEX(8)] = spec_rates[INDEX(7)] * (1.7007339999999999e+01 / rho);
  dy[INDEX(9)] = spec_rates[INDEX(8)] * (3.4014679999999998e+01 / rho);
  dy[INDEX(10)] = spec_rates[INDEX(9)] * (3.3006740000000001e+01 / rho);
  dy[INDEX(11)] = spec_rates[INDEX(10)] * (2.8010399999999997e+01 / rho);
  dy[INDEX(12)] = spec_rates[INDEX(11)] * (4.4009799999999998e+01 / rho);
  dy[INDEX(13)] = spec_rates[INDEX(12)] * (4.5017739999999996e+01 / rho);
  dy[INDEX(14)] = spec_rates[INDEX(13)] * (1.6042760000000001e+01 / rho);
  dy[INDEX(15)] = spec_rates[INDEX(14)] * (1.5034820000000000e+01 / rho);
  dy[INDEX(16)] = spec_rates[INDEX(15)] * (1.4026879999999998e+01 / rho);
  dy[INDEX(17)] = spec_rates[INDEX(16)] * (1.4026879999999998e+01 / rho);
  dy[INDEX(18)] = spec_rates[INDEX(17)] * (1.2010999999999999e+01 / rho);
  dy[INDEX(19)] = spec_rates[INDEX(18)] * (1.3018939999999999e+01 / rho);
  dy[INDEX(20)] = spec_rates[INDEX(19)] * (4.8041560000000004e+01 / rho);
  dy[INDEX(21)] = spec_rates[INDEX(20)] * (4.7033619999999999e+01 / rho);
  dy[INDEX(22)] = spec_rates[INDEX(21)] * (3.2042160000000003e+01 / rho);
  dy[INDEX(23)] = spec_rates[INDEX(22)] * (3.1034219999999998e+01 / rho);
  dy[INDEX(24)] = spec_rates[INDEX(23)] * (3.1034219999999998e+01 / rho);
  dy[INDEX(25)] = spec_rates[INDEX(24)] * (3.0026280000000000e+01 / rho);
  dy[INDEX(26)] = spec_rates[INDEX(25)] * (2.9018339999999998e+01 / rho);
  dy[INDEX(27)] = spec_rates[INDEX(26)] * (6.2025079999999996e+01 / rho);
  dy[INDEX(28)] = spec_rates[INDEX(27)] * (4.6025679999999994e+01 / rho);
  dy[INDEX(29)] = spec_rates[INDEX(28)] * (4.5017739999999996e+01 / rho);
  dy[INDEX(30)] = spec_rates[INDEX(29)] * (3.0069640000000000e+01 / rho);
  dy[INDEX(31)] = spec_rates[INDEX(30)] * (2.9061699999999998e+01 / rho);
  dy[INDEX(32)] = spec_rates[INDEX(31)] * (6.2068439999999995e+01 / rho);
  dy[INDEX(33)] = spec_rates[INDEX(32)] * (6.1060499999999998e+01 / rho);
  dy[INDEX(34)] = spec_rates[INDEX(33)] * (2.8053759999999997e+01 / rho);
  dy[INDEX(35)] = spec_rates[INDEX(34)] * (2.7045819999999999e+01 / rho);
  dy[INDEX(36)] = spec_rates[INDEX(35)] * (2.6037879999999998e+01 / rho);
  dy[INDEX(37)] = spec_rates[INDEX(36)] * (2.5029940000000000e+01 / rho);
  dy[INDEX(38)] = spec_rates[INDEX(37)] * (4.6069040000000001e+01 / rho);
  dy[INDEX(39)] = spec_rates[INDEX(38)] * (4.5061099999999996e+01 / rho);
  dy[INDEX(40)] = spec_rates[INDEX(39)] * (4.5061099999999996e+01 / rho);
  dy[INDEX(41)] = spec_rates[INDEX(40)] * (4.5061099999999996e+01 / rho);
  dy[INDEX(42)] = spec_rates[INDEX(41)] * (6.1060499999999998e+01 / rho);
  dy[INDEX(43)] = spec_rates[INDEX(42)] * (4.4053159999999998e+01 / rho);
  dy[INDEX(44)] = spec_rates[INDEX(43)] * (4.3045220000000000e+01 / rho);
  dy[INDEX(45)] = spec_rates[INDEX(44)] * (4.4053159999999998e+01 / rho);
  dy[INDEX(46)] = spec_rates[INDEX(45)] * (4.3045220000000000e+01 / rho);
  dy[INDEX(47)] = spec_rates[INDEX(46)] * (4.3045220000000000e+01 / rho);
  dy[INDEX(48)] = spec_rates[INDEX(47)] * (4.2037279999999996e+01 / rho);
  dy[INDEX(49)] = spec_rates[INDEX(48)] * (4.1029339999999998e+01 / rho);
  dy[INDEX(50)] = spec_rates[INDEX(49)] * (4.6025679999999994e+01 / rho);
  dy[INDEX(51)] = spec_rates[INDEX(50)] * (7.5044019999999989e+01 / rho);
  dy[INDEX(52)] = spec_rates[INDEX(51)] * (7.6051959999999994e+01 / rho);
  dy[INDEX(53)] = spec_rates[INDEX(52)] * (6.0052560000000000e+01 / rho);
  dy[INDEX(54)] = spec_rates[INDEX(53)] * (5.8036679999999997e+01 / rho);
  dy[INDEX(55)] = spec_rates[INDEX(54)] * (9.3059299999999993e+01 / rho);
  dy[INDEX(56)] = spec_rates[INDEX(55)] * (7.6051959999999994e+01 / rho);
  dy[INDEX(57)] = spec_rates[INDEX(56)] * (6.0052560000000000e+01 / rho);
  dy[INDEX(58)] = spec_rates[INDEX(57)] * (5.9044619999999995e+01 / rho);
  dy[INDEX(59)] = spec_rates[INDEX(58)] * (4.4096519999999998e+01 / rho);
  dy[INDEX(60)] = spec_rates[INDEX(59)] * (4.3088580000000000e+01 / rho);
  dy[INDEX(61)] = spec_rates[INDEX(60)] * (4.3088580000000000e+01 / rho);
  dy[INDEX(62)] = spec_rates[INDEX(61)] * (4.2080640000000002e+01 / rho);
  dy[INDEX(63)] = spec_rates[INDEX(62)] * (4.1072699999999998e+01 / rho);
  dy[INDEX(64)] = spec_rates[INDEX(63)] * (4.1072699999999998e+01 / rho);
  dy[INDEX(65)] = spec_rates[INDEX(64)] * (4.1072699999999998e+01 / rho);
  dy[INDEX(66)] = spec_rates[INDEX(65)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(67)] = spec_rates[INDEX(66)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(68)] = spec_rates[INDEX(67)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(69)] = spec_rates[INDEX(68)] * (8.8106319999999997e+01 / rho);
  dy[INDEX(70)] = spec_rates[INDEX(69)] * (5.6064160000000001e+01 / rho);
  dy[INDEX(71)] = spec_rates[INDEX(70)] * (7.4079440000000005e+01 / rho);
  dy[INDEX(72)] = spec_rates[INDEX(71)] * (5.9087980000000002e+01 / rho);
  dy[INDEX(73)] = spec_rates[INDEX(72)] * (5.9087980000000002e+01 / rho);
  dy[INDEX(74)] = spec_rates[INDEX(73)] * (9.1086780000000005e+01 / rho);
  dy[INDEX(75)] = spec_rates[INDEX(74)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(76)] = spec_rates[INDEX(75)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(77)] = spec_rates[INDEX(76)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(78)] = spec_rates[INDEX(77)] * (4.0064760000000000e+01 / rho);
  dy[INDEX(79)] = spec_rates[INDEX(78)] * (4.0064760000000000e+01 / rho);
  dy[INDEX(80)] = spec_rates[INDEX(79)] * (3.9056820000000002e+01 / rho);
  dy[INDEX(81)] = spec_rates[INDEX(80)] * (3.8048880000000004e+01 / rho);
  dy[INDEX(82)] = spec_rates[INDEX(81)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(83)] = spec_rates[INDEX(82)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(84)] = spec_rates[INDEX(83)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(85)] = spec_rates[INDEX(84)] * (5.6064160000000001e+01 / rho);
  dy[INDEX(86)] = spec_rates[INDEX(85)] * (5.8080040000000004e+01 / rho);
  dy[INDEX(87)] = spec_rates[INDEX(86)] * (5.7072099999999999e+01 / rho);
  dy[INDEX(88)] = spec_rates[INDEX(87)] * (5.8123399999999997e+01 / rho);
  dy[INDEX(89)] = spec_rates[INDEX(88)] * (5.7115459999999999e+01 / rho);
  dy[INDEX(90)] = spec_rates[INDEX(89)] * (5.7115459999999999e+01 / rho);
  dy[INDEX(91)] = spec_rates[INDEX(90)] * (5.8123399999999997e+01 / rho);
  dy[INDEX(92)] = spec_rates[INDEX(91)] * (5.7115459999999999e+01 / rho);
  dy[INDEX(93)] = spec_rates[INDEX(92)] * (5.7115459999999999e+01 / rho);
  dy[INDEX(94)] = spec_rates[INDEX(93)] * (5.6107519999999994e+01 / rho);
  dy[INDEX(95)] = spec_rates[INDEX(94)] * (5.5099579999999996e+01 / rho);
  dy[INDEX(96)] = spec_rates[INDEX(95)] * (7.1098979999999997e+01 / rho);
  dy[INDEX(97)] = spec_rates[INDEX(96)] * (5.6107519999999994e+01 / rho);
  dy[INDEX(98)] = spec_rates[INDEX(97)] * (5.6107519999999994e+01 / rho);
  dy[INDEX(99)] = spec_rates[INDEX(98)] * (5.5099579999999996e+01 / rho);
  dy[INDEX(100)] = spec_rates[INDEX(99)] * (5.5099579999999996e+01 / rho);
  dy[INDEX(101)] = spec_rates[INDEX(100)] * (7.1098979999999997e+01 / rho);
  dy[INDEX(102)] = spec_rates[INDEX(101)] * (5.4091639999999998e+01 / rho);
  dy[INDEX(103)] = spec_rates[INDEX(102)] * (5.3083699999999993e+01 / rho);
  dy[INDEX(104)] = spec_rates[INDEX(103)] * (5.2075759999999995e+01 / rho);
  dy[INDEX(105)] = spec_rates[INDEX(104)] * (5.1067819999999998e+01 / rho);
  dy[INDEX(106)] = spec_rates[INDEX(105)] * (5.0059880000000000e+01 / rho);
  dy[INDEX(107)] = spec_rates[INDEX(106)] * (7.8113640000000004e+01 / rho);
  dy[INDEX(108)] = spec_rates[INDEX(107)] * (7.8113640000000004e+01 / rho);
  dy[INDEX(109)] = spec_rates[INDEX(108)] * (7.7105699999999999e+01 / rho);
  dy[INDEX(110)] = spec_rates[INDEX(109)] * (6.6102639999999994e+01 / rho);
  dy[INDEX(111)] = spec_rates[INDEX(110)] * (6.5094699999999989e+01 / rho);
  dy[INDEX(112)] = spec_rates[INDEX(111)] * (8.0129519999999999e+01 / rho);
  dy[INDEX(113)] = spec_rates[INDEX(112)] * (1.2817352000000000e+02 / rho);
  dy[INDEX(114)] = spec_rates[INDEX(113)] * (3.0006140000000002e+01 / rho);
  dy[INDEX(115)] = spec_rates[INDEX(114)] * (4.4012880000000003e+01 / rho);
  dy[INDEX(116)] = spec_rates[INDEX(115)] * (4.6005539999999996e+01 / rho);
  dy[INDEX(117)] = spec_rates[INDEX(116)] * (3.1014080000000000e+01 / rho);
  dy[INDEX(118)] = spec_rates[INDEX(117)] * (4.7013480000000001e+01 / rho);
  dy[INDEX(119)] = spec_rates[INDEX(118)] * (4.7013480000000001e+01 / rho);
  dy[INDEX(120)] = spec_rates[INDEX(119)] * (6.3012879999999996e+01 / rho);
  dy[INDEX(121)] = spec_rates[INDEX(120)] * (3.0029360000000000e+01 / rho);
  dy[INDEX(122)] = spec_rates[INDEX(121)] * (3.0029360000000000e+01 / rho);
  dy[INDEX(123)] = spec_rates[INDEX(122)] * (4.5020820000000001e+01 / rho);
  dy[INDEX(124)] = spec_rates[INDEX(123)] * (4.6028759999999998e+01 / rho);
  dy[INDEX(125)] = spec_rates[INDEX(124)] * (3.3029960000000003e+01 / rho);
  dy[INDEX(126)] = spec_rates[INDEX(125)] * (3.2022019999999998e+01 / rho);
  dy[INDEX(127)] = spec_rates[INDEX(126)] * (1.7030560000000001e+01 / rho);
  dy[INDEX(128)] = spec_rates[INDEX(127)] * (3.2045240000000000e+01 / rho);
  dy[INDEX(129)] = spec_rates[INDEX(128)] * (1.4006740000000001e+01 / rho);
  dy[INDEX(130)] = spec_rates[INDEX(129)] * (6.2004939999999998e+01 / rho);
  dy[INDEX(131)] = spec_rates[INDEX(130)] * (1.5014680000000000e+01 / rho);
  dy[INDEX(132)] = spec_rates[INDEX(131)] * (2.9021420000000003e+01 / rho);
  dy[INDEX(133)] = spec_rates[INDEX(132)] * (1.6022620000000000e+01 / rho);
  dy[INDEX(134)] = spec_rates[INDEX(133)] * (3.2022019999999998e+01 / rho);
  dy[INDEX(135)] = spec_rates[INDEX(134)] * (3.1037300000000002e+01 / rho);
  dy[INDEX(136)] = spec_rates[INDEX(135)] * (2.7025680000000001e+01 / rho);
  dy[INDEX(137)] = spec_rates[INDEX(136)] * (2.7025680000000001e+01 / rho);
  dy[INDEX(138)] = spec_rates[INDEX(137)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(139)] = spec_rates[INDEX(138)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(140)] = spec_rates[INDEX(139)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(141)] = spec_rates[INDEX(140)] * (4.4033020000000000e+01 / rho);
  dy[INDEX(142)] = spec_rates[INDEX(141)] * (4.5040959999999998e+01 / rho);
  dy[INDEX(143)] = spec_rates[INDEX(142)] * (6.1040360000000000e+01 / rho);
  dy[INDEX(144)] = spec_rates[INDEX(143)] * (6.1040360000000000e+01 / rho);
  dy[INDEX(145)] = spec_rates[INDEX(144)] * (7.7039760000000001e+01 / rho);
  dy[INDEX(146)] = spec_rates[INDEX(145)] * (4.1052560000000000e+01 / rho);
  dy[INDEX(147)] = spec_rates[INDEX(146)] * (2.6017740000000000e+01 / rho);
  dy[INDEX(148)] = spec_rates[INDEX(147)] * (4.0024479999999997e+01 / rho);
  dy[INDEX(149)] = spec_rates[INDEX(148)] * (4.2017139999999998e+01 / rho);
  dy[INDEX(150)] = spec_rates[INDEX(149)] * (4.1032420000000002e+01 / rho);
  dy[INDEX(151)] = spec_rates[INDEX(150)] * (2.8033619999999999e+01 / rho);
  dy[INDEX(152)] = spec_rates[INDEX(151)] * (2.8033619999999999e+01 / rho);
  dy[INDEX(153)] = spec_rates[INDEX(152)] * (5.2035480000000000e+01 / rho);
  dy[INDEX(154)] = spec_rates[INDEX(153)] * (4.0044619999999995e+01 / rho);
  dy[INDEX(155)] = spec_rates[INDEX(154)] * (2.9041560000000000e+01 / rho);
  dy[INDEX(156)] = spec_rates[INDEX(155)] * (3.1057440000000000e+01 / rho);
  dy[INDEX(157)] = spec_rates[INDEX(156)] * (3.0049500000000002e+01 / rho);
  dy[INDEX(158)] = spec_rates[INDEX(157)] * (3.0049500000000002e+01 / rho);

} // end dydt

#endif
