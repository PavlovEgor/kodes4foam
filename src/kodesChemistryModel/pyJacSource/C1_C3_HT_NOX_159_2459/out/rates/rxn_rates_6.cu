#include "rates.cuh"
__device__ void eval_rxn_rates_6 (const double T, const double pres, const double * __restrict__ C, double * __restrict__ fwd_rxn_rates, double * __restrict__ rev_rxn_rates) {
  extern volatile __shared__ double shared_temp[];
  register double logT = log(T);

  register double kf;
  register double Kc;

  //rxn 1500
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(52)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(10)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(23)];
  shared_temp[threadIdx.x] = C[INDEX(58)];
  fwd_rxn_rates[INDEX(1500)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(59)] * exp(3.9889840465642745e+00 + 2.0 * logT - (4.1185753676115746e+03 / T));
  //rxn 1501
  fwd_rxn_rates[INDEX(1501)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(63)] * exp(4.6821312271242199e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1502
  fwd_rxn_rates[INDEX(1502)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(64)] * exp(4.6821312271242199e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1503
  fwd_rxn_rates[INDEX(1503)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(62)] * exp(5.3734248390348425e+00 + 2.0 * logT - (5.5215313856563262e+03 / T));
  //rxn 1504
  fwd_rxn_rates[INDEX(1504)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(102)] * exp(4.6821312271242199e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1505
  fwd_rxn_rates[INDEX(1505)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(99)] * exp(3.9889840465642745e+00 + 2.0 * logT - (2.5326688445666259e+03 / T));
  //rxn 1506
  fwd_rxn_rates[INDEX(1506)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(98)] * exp(4.9126548857360524e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1507
  fwd_rxn_rates[INDEX(1507)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(94)] * exp(4.6821312271242199e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1508
  fwd_rxn_rates[INDEX(1508)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(104)] * exp(5.0974244241686471e+00 + 2.0 * logT - (1.8646197112378445e+03 / T));
  //rxn 1509
  fwd_rxn_rates[INDEX(1509)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(108)] * exp(1.1904967552746252e+01 + 1.0 * logT - (4.3671909936302438e+02 / T));
  //rxn 1510
  fwd_rxn_rates[INDEX(1510)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(110)] * exp(4.9126548857360524e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1511
  fwd_rxn_rates[INDEX(1511)] = C[INDEX(5)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(9.1484649682580947e+00 + 2.0 * logT - (5.8842470158111621e+02 / T));
  //rxn 1512
  fwd_rxn_rates[INDEX(1512)] = C[INDEX(4)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.8246778911641979e+00 + 2.0 * logT - (2.0836253948406356e+04 / T));
  //rxn 1513
  fwd_rxn_rates[INDEX(1513)] = C[INDEX(7)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(1.5607270027192330e+01 + 1.0 * logT - (-4.3030304991107010e+02 / T));
  //rxn 1514
  fwd_rxn_rates[INDEX(1514)] = C[INDEX(9)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.0709060887878188e+00 + 2.0 * logT - (7.0767213864622709e+03 / T));
  //rxn 1515
  fwd_rxn_rates[INDEX(1515)] = C[INDEX(22)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.1428324637076415e+00 + 2.0 * logT - (1.2162816882645963e+02 / T));
  //rxn 1516
  fwd_rxn_rates[INDEX(1516)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.7291561657690826e+00 + 2.0 * logT - (4.2633617008128531e+03 / T));
  //rxn 1517
  fwd_rxn_rates[INDEX(1517)] = C[INDEX(25)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.2225762680713688e+00 + 2.0 * logT - (5.2791506500462619e+03 / T));
  //rxn 1518
  fwd_rxn_rates[INDEX(1518)] = C[INDEX(45)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.2915691395583204e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1519
  fwd_rxn_rates[INDEX(1519)] = C[INDEX(46)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.5217885770490405e+00 + 2.0 * logT - (4.8917420358437867e+03 / T));
  //rxn 1520
  fwd_rxn_rates[INDEX(1520)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(79)] * exp(4.6821312271242199e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1521
  fwd_rxn_rates[INDEX(1521)] = C[INDEX(48)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.1428324637076415e+00 + 2.0 * logT - (1.8646197112378445e+03 / T));
  //rxn 1522
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(53)];
  shared_temp[threadIdx.x] = C[INDEX(3)];
  fwd_rxn_rates[INDEX(1522)] = shared_temp[threadIdx.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.0063675676502459e+00 + 2.0 * logT - (2.0128782594366510e+03 / T));
  //rxn 1523
  fwd_rxn_rates[INDEX(1523)] = C[INDEX(14)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.4011973816621555e+00 + 2.0 * logT - (2.5160978242958136e+03 / T));
  //rxn 1524
  fwd_rxn_rates[INDEX(1524)] = C[INDEX(34)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.2165621949463494e+00 + 2.0 * logT - (1.9625563029507348e+03 / T));
  //rxn 1525
  fwd_rxn_rates[INDEX(1525)] = C[INDEX(30)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(2.9957322735539909e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1526
  fwd_rxn_rates[INDEX(1526)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(60)] * exp(2.6026896854443837e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1527
  fwd_rxn_rates[INDEX(1527)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(59)] * exp(2.6026896854443837e+00 + 2.0 * logT - (4.9818736921057116e+03 / T));
  //rxn 1528
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1528)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(63)] * exp(3.2958368660043291e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1529
  fwd_rxn_rates[INDEX(1529)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(64)] * exp(3.2958368660043291e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1530
  fwd_rxn_rates[INDEX(1530)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(62)] * exp(3.9871304779149512e+00 + 2.0 * logT - (6.4412104301972831e+03 / T));
  //rxn 1531
  fwd_rxn_rates[INDEX(1531)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(102)] * exp(3.2958368660043291e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1532
  fwd_rxn_rates[INDEX(1532)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(99)] * exp(2.6026896854443837e+00 + 2.0 * logT - (3.3212491280704739e+03 / T));
  //rxn 1533
  fwd_rxn_rates[INDEX(1533)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(98)] * exp(3.5263605246161616e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1534
  fwd_rxn_rates[INDEX(1534)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(94)] * exp(3.2958368660043291e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1535
  fwd_rxn_rates[INDEX(1535)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(104)] * exp(3.7111300630487558e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1536
  fwd_rxn_rates[INDEX(1536)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(108)] * exp(1.0518673191626361e+01 + 1.0 * logT - (1.0064391297183255e+03 / T));
  //rxn 1537
  fwd_rxn_rates[INDEX(1537)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(110)] * exp(3.5263605246161616e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1538
  fwd_rxn_rates[INDEX(1538)] = C[INDEX(5)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.7621706071382048e+00 + 2.0 * logT - (1.2580489121479068e+03 / T));
  //rxn 1539
  fwd_rxn_rates[INDEX(1539)] = C[INDEX(4)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.4383835300443071e+00 + 2.0 * logT - (2.2141660853803161e+04 / T));
  //rxn 1540
  fwd_rxn_rates[INDEX(1540)] = C[INDEX(7)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(1.4220975666072439e+01 + 1.0 * logT - (2.5160978242958137e+01 / T));
  //rxn 1541
  fwd_rxn_rates[INDEX(1541)] = C[INDEX(9)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.6846117276679271e+00 + 2.0 * logT - (8.0515130377466039e+03 / T));
  //rxn 1542
  fwd_rxn_rates[INDEX(1542)] = C[INDEX(22)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.7565381025877511e+00 + 2.0 * logT - (7.5482934728874409e+02 / T));
  //rxn 1543
  fwd_rxn_rates[INDEX(1543)] = C[INDEX(23)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.3428618046491918e+00 + 2.0 * logT - (5.1328395615634599e+03 / T));
  //rxn 1544
  fwd_rxn_rates[INDEX(1544)] = C[INDEX(25)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.8362819069514780e+00 + 2.0 * logT - (6.1896006477677020e+03 / T));
  //rxn 1545
  fwd_rxn_rates[INDEX(1545)] = C[INDEX(45)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.9052747784384296e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1546
  fwd_rxn_rates[INDEX(1546)] = C[INDEX(46)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.1354942159291497e+00 + 2.0 * logT - (5.7870249958803715e+03 / T));
  //rxn 1547
  fwd_rxn_rates[INDEX(1547)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(79)] * exp(3.2958368660043291e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1548
  fwd_rxn_rates[INDEX(1548)] = C[INDEX(48)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.7565381025877511e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1549
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(10)];
  fwd_rxn_rates[INDEX(1549)] = shared_temp[threadIdx.x] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(1.0085809109330082e+01 + 2.0 * logT - (1.2950909043171898e+03 / T));
  //rxn 1550
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(25)];
  fwd_rxn_rates[INDEX(1550)] = C[INDEX(14)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.4806389233419912e+00 + 2.0 * logT - (1.7694810203055713e+03 / T));
  //rxn 1551
  fwd_rxn_rates[INDEX(1551)] = C[INDEX(34)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.2960037366261856e+00 + 2.0 * logT - (1.2477832330247800e+03 / T));
  //rxn 1552
  fwd_rxn_rates[INDEX(1552)] = C[INDEX(30)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.0751738152338266e+00 + 2.0 * logT - (2.5805149607934354e+03 / T));
  //rxn 1553
  fwd_rxn_rates[INDEX(1553)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(60)] * exp(4.6821312271242199e+00 + 2.0 * logT - (2.5805149607934354e+03 / T));
  //rxn 1554
  fwd_rxn_rates[INDEX(1554)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(59)] * exp(4.6821312271242199e+00 + 2.0 * logT - (4.1185753676115746e+03 / T));
  //rxn 1555
  shared_temp[threadIdx.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1555)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(63)] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1556
  fwd_rxn_rates[INDEX(1556)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(64)] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1557
  fwd_rxn_rates[INDEX(1557)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(62)] * exp(6.0665720195947870e+00 + 2.0 * logT - (5.5215313856563262e+03 / T));
  //rxn 1558
  fwd_rxn_rates[INDEX(1558)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(102)] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1559
  fwd_rxn_rates[INDEX(1559)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(99)] * exp(4.6821312271242199e+00 + 2.0 * logT - (2.5326688445666259e+03 / T));
  //rxn 1560
  fwd_rxn_rates[INDEX(1560)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(98)] * exp(5.6058020662959978e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1561
  fwd_rxn_rates[INDEX(1561)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(94)] * exp(5.3752784076841653e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1562
  fwd_rxn_rates[INDEX(1562)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(104)] * exp(5.7905716047285916e+00 + 2.0 * logT - (1.8646197112378445e+03 / T));
  //rxn 1563
  fwd_rxn_rates[INDEX(1563)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(108)] * exp(1.2598114733306197e+01 + 1.0 * logT - (4.3671909936302438e+02 / T));
  //rxn 1564
  fwd_rxn_rates[INDEX(1564)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(110)] * exp(5.6058020662959978e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1565
  fwd_rxn_rates[INDEX(1565)] = C[INDEX(5)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(9.8416121488180401e+00 + 2.0 * logT - (5.8842470158111621e+02 / T));
  //rxn 1566
  fwd_rxn_rates[INDEX(1566)] = C[INDEX(4)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(9.5178250717241433e+00 + 2.0 * logT - (2.0836253948406356e+04 / T));
  //rxn 1567
  fwd_rxn_rates[INDEX(1567)] = C[INDEX(7)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(1.6300417207752275e+01 + 1.0 * logT - (-4.3030304991107010e+02 / T));
  //rxn 1568
  fwd_rxn_rates[INDEX(1568)] = C[INDEX(9)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(8.7640532693477624e+00 + 2.0 * logT - (7.0767213864622709e+03 / T));
  //rxn 1569
  fwd_rxn_rates[INDEX(1569)] = C[INDEX(22)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.8359796442675869e+00 + 2.0 * logT - (1.2162816882645963e+02 / T));
  //rxn 1570
  fwd_rxn_rates[INDEX(1570)] = C[INDEX(23)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.4223033463290280e+00 + 2.0 * logT - (4.2633617008128531e+03 / T));
  //rxn 1571
  fwd_rxn_rates[INDEX(1571)] = shared_temp[threadIdx.x + 3 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.9157234486313142e+00 + 2.0 * logT - (5.2791506500462619e+03 / T));
  //rxn 1572
  fwd_rxn_rates[INDEX(1572)] = C[INDEX(45)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.9847163201182658e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1573
  fwd_rxn_rates[INDEX(1573)] = C[INDEX(46)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.2149357576089859e+00 + 2.0 * logT - (4.8917420358437867e+03 / T));
  //rxn 1574
  fwd_rxn_rates[INDEX(1574)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(79)] * exp(5.3752784076841653e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1575
  fwd_rxn_rates[INDEX(1575)] = C[INDEX(48)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.8359796442675869e+00 + 2.0 * logT - (1.8646197112378445e+03 / T));
  //rxn 1576
  shared_temp[threadIdx.x] = C[INDEX(49)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(7)];
  fwd_rxn_rates[INDEX(1576)] = C[INDEX(3)] * shared_temp[threadIdx.x] * exp(7.3132203870903014e+00 + 2.0 * logT - (3.1951121119428767e+03 / T));
  //rxn 1577
  fwd_rxn_rates[INDEX(1577)] = C[INDEX(14)] * shared_temp[threadIdx.x] * exp(2.7080502011022101e+00 + 2.0 * logT - (3.7458154749421496e+03 / T));
  //rxn 1578
  fwd_rxn_rates[INDEX(1578)] = C[INDEX(34)] * shared_temp[threadIdx.x] * exp(3.5234150143864045e+00 + 2.0 * logT - (3.1398284105474486e+03 / T));
  //rxn 1579
  fwd_rxn_rates[INDEX(1579)] = C[INDEX(30)] * shared_temp[threadIdx.x] * exp(2.3025850929940459e+00 + 2.0 * logT - (4.6744870531071401e+03 / T));
  //rxn 1580
  fwd_rxn_rates[INDEX(1580)] = shared_temp[threadIdx.x] * C[INDEX(60)] * exp(1.9095425048844386e+00 + 2.0 * logT - (4.6744870531071401e+03 / T));
  //rxn 1581
  fwd_rxn_rates[INDEX(1581)] = shared_temp[threadIdx.x] * C[INDEX(59)] * exp(1.9095425048844386e+00 + 2.0 * logT - (6.4037809589630588e+03 / T));
  //rxn 1582
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1582)] = shared_temp[threadIdx.x] * C[INDEX(63)] * exp(2.6026896854443837e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1583
  fwd_rxn_rates[INDEX(1583)] = shared_temp[threadIdx.x] * C[INDEX(64)] * exp(2.6026896854443837e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1584
  fwd_rxn_rates[INDEX(1584)] = shared_temp[threadIdx.x] * C[INDEX(62)] * exp(3.2939832973550063e+00 + 2.0 * logT - (7.9559818355537409e+03 / T));
  //rxn 1585
  fwd_rxn_rates[INDEX(1585)] = shared_temp[threadIdx.x] * C[INDEX(102)] * exp(2.6026896854443837e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1586
  fwd_rxn_rates[INDEX(1586)] = shared_temp[threadIdx.x] * C[INDEX(99)] * exp(1.9095425048844386e+00 + 2.0 * logT - (4.6200839859502157e+03 / T));
  //rxn 1587
  fwd_rxn_rates[INDEX(1587)] = shared_temp[threadIdx.x] * C[INDEX(98)] * exp(2.8332133440562162e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1588
  fwd_rxn_rates[INDEX(1588)] = shared_temp[threadIdx.x] * C[INDEX(94)] * exp(2.6026896854443837e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1589
  fwd_rxn_rates[INDEX(1589)] = shared_temp[threadIdx.x] * C[INDEX(104)] * exp(3.0179828824888109e+00 + 2.0 * logT - (3.8555274044727444e+03 / T));
  //rxn 1590
  fwd_rxn_rates[INDEX(1590)] = shared_temp[threadIdx.x] * C[INDEX(108)] * exp(9.8255260110664153e+00 + 1.0 * logT - (1.9448027167025034e+03 / T));
  //rxn 1591
  fwd_rxn_rates[INDEX(1591)] = shared_temp[threadIdx.x] * C[INDEX(110)] * exp(2.8332133440562162e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1592
  fwd_rxn_rates[INDEX(1592)] = C[INDEX(5)] * shared_temp[threadIdx.x] * exp(7.0690234265782594e+00 + 2.0 * logT - (2.3609552324497336e+03 / T));
  //rxn 1593
  fwd_rxn_rates[INDEX(1593)] = C[INDEX(4)] * shared_temp[threadIdx.x] * exp(6.7452363494843626e+00 + 2.0 * logT - (2.4291741927598665e+04 / T));
  //rxn 1594
  fwd_rxn_rates[INDEX(1594)] = shared_temp[threadIdx.x + 3 * blockDim.x] * shared_temp[threadIdx.x] * exp(1.3527828485512494e+01 + 1.0 * logT - (7.7533554455675505e+02 / T));
  //rxn 1595
  fwd_rxn_rates[INDEX(1595)] = C[INDEX(9)] * shared_temp[threadIdx.x] * exp(5.9914645471079817e+00 + 2.0 * logT - (9.6570501560167086e+03 / T));
  //rxn 1596
  fwd_rxn_rates[INDEX(1596)] = C[INDEX(22)] * shared_temp[threadIdx.x] * exp(3.0633909220278057e+00 + 2.0 * logT - (1.7977518954593588e+03 / T));
  //rxn 1597
  fwd_rxn_rates[INDEX(1597)] = C[INDEX(23)] * shared_temp[threadIdx.x] * exp(2.6497146240892469e+00 + 2.0 * logT - (6.5649219280222596e+03 / T));
  //rxn 1598
  fwd_rxn_rates[INDEX(1598)] = C[INDEX(25)] * shared_temp[threadIdx.x] * exp(4.1431347263915326e+00 + 2.0 * logT - (7.6891697900697645e+03 / T));
  //rxn 1599
  fwd_rxn_rates[INDEX(1599)] = C[INDEX(45)] * shared_temp[threadIdx.x] * exp(4.2121275978784842e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1600
  fwd_rxn_rates[INDEX(1600)] = C[INDEX(46)] * shared_temp[threadIdx.x] * exp(2.4423470353692043e+00 + 2.0 * logT - (7.2616092867871757e+03 / T));
  //rxn 1601
  fwd_rxn_rates[INDEX(1601)] = shared_temp[threadIdx.x] * C[INDEX(79)] * exp(2.6026896854443837e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1602
  fwd_rxn_rates[INDEX(1602)] = C[INDEX(48)] * shared_temp[threadIdx.x] * exp(3.0633909220278057e+00 + 2.0 * logT - (3.8555274044727444e+03 / T));
  //rxn 1603
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(3)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(11)];
  fwd_rxn_rates[INDEX(1603)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x] * exp(7.3132203870903014e+00 + 2.0 * logT - (3.1951121119428767e+03 / T));
  //rxn 1604
  fwd_rxn_rates[INDEX(1604)] = C[INDEX(14)] * shared_temp[threadIdx.x] * exp(2.7080502011022101e+00 + 2.0 * logT - (3.7458154749421496e+03 / T));
  //rxn 1605
  fwd_rxn_rates[INDEX(1605)] = C[INDEX(34)] * shared_temp[threadIdx.x] * exp(3.5234150143864045e+00 + 2.0 * logT - (3.1398284105474486e+03 / T));
  //rxn 1606
  fwd_rxn_rates[INDEX(1606)] = C[INDEX(30)] * shared_temp[threadIdx.x] * exp(2.3025850929940459e+00 + 2.0 * logT - (4.6744870531071401e+03 / T));
  //rxn 1607
  fwd_rxn_rates[INDEX(1607)] = shared_temp[threadIdx.x] * C[INDEX(60)] * exp(1.9095425048844386e+00 + 2.0 * logT - (4.6744870531071401e+03 / T));
  //rxn 1608
  fwd_rxn_rates[INDEX(1608)] = shared_temp[threadIdx.x] * C[INDEX(59)] * exp(1.9095425048844386e+00 + 2.0 * logT - (6.4037809589630588e+03 / T));
  //rxn 1609
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1609)] = shared_temp[threadIdx.x] * C[INDEX(63)] * exp(2.6026896854443837e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1610
  fwd_rxn_rates[INDEX(1610)] = shared_temp[threadIdx.x] * C[INDEX(64)] * exp(2.6026896854443837e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1611
  fwd_rxn_rates[INDEX(1611)] = shared_temp[threadIdx.x] * C[INDEX(62)] * exp(3.2939832973550063e+00 + 2.0 * logT - (7.9559818355537409e+03 / T));
  //rxn 1612
  fwd_rxn_rates[INDEX(1612)] = shared_temp[threadIdx.x] * C[INDEX(102)] * exp(2.6026896854443837e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1613
  fwd_rxn_rates[INDEX(1613)] = shared_temp[threadIdx.x] * C[INDEX(99)] * exp(1.9095425048844386e+00 + 2.0 * logT - (4.6200839859502157e+03 / T));
  //rxn 1614
  fwd_rxn_rates[INDEX(1614)] = shared_temp[threadIdx.x] * C[INDEX(98)] * exp(2.8332133440562162e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1615
  fwd_rxn_rates[INDEX(1615)] = shared_temp[threadIdx.x] * C[INDEX(94)] * exp(2.6026896854443837e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1616
  fwd_rxn_rates[INDEX(1616)] = shared_temp[threadIdx.x] * C[INDEX(104)] * exp(3.0179828824888109e+00 + 2.0 * logT - (3.8555274044727444e+03 / T));
  //rxn 1617
  fwd_rxn_rates[INDEX(1617)] = shared_temp[threadIdx.x] * C[INDEX(108)] * exp(9.8255260110664153e+00 + 1.0 * logT - (1.9448027167025034e+03 / T));
  //rxn 1618
  fwd_rxn_rates[INDEX(1618)] = shared_temp[threadIdx.x] * C[INDEX(110)] * exp(2.8332133440562162e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1619
  fwd_rxn_rates[INDEX(1619)] = C[INDEX(5)] * shared_temp[threadIdx.x] * exp(7.0690234265782594e+00 + 2.0 * logT - (2.3609552324497336e+03 / T));
  //rxn 1620
  fwd_rxn_rates[INDEX(1620)] = C[INDEX(4)] * shared_temp[threadIdx.x] * exp(6.7452363494843626e+00 + 2.0 * logT - (2.4291741927598665e+04 / T));
  //rxn 1621
  fwd_rxn_rates[INDEX(1621)] = C[INDEX(7)] * shared_temp[threadIdx.x] * exp(1.3527828485512494e+01 + 1.0 * logT - (7.7533554455675505e+02 / T));
  //rxn 1622
  fwd_rxn_rates[INDEX(1622)] = C[INDEX(9)] * shared_temp[threadIdx.x] * exp(5.9914645471079817e+00 + 2.0 * logT - (9.6570501560167086e+03 / T));
  //rxn 1623
  fwd_rxn_rates[INDEX(1623)] = C[INDEX(22)] * shared_temp[threadIdx.x] * exp(3.0633909220278057e+00 + 2.0 * logT - (1.7977518954593588e+03 / T));
  //rxn 1624
  fwd_rxn_rates[INDEX(1624)] = C[INDEX(23)] * shared_temp[threadIdx.x] * exp(2.6497146240892469e+00 + 2.0 * logT - (6.5649219280222596e+03 / T));
  //rxn 1625
  fwd_rxn_rates[INDEX(1625)] = C[INDEX(25)] * shared_temp[threadIdx.x] * exp(4.1431347263915326e+00 + 2.0 * logT - (7.6891697900697645e+03 / T));
  //rxn 1626
  fwd_rxn_rates[INDEX(1626)] = C[INDEX(45)] * shared_temp[threadIdx.x] * exp(4.2121275978784842e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1627
  fwd_rxn_rates[INDEX(1627)] = C[INDEX(46)] * shared_temp[threadIdx.x] * exp(2.4423470353692043e+00 + 2.0 * logT - (7.2616092867871757e+03 / T));
  //rxn 1628
  fwd_rxn_rates[INDEX(1628)] = shared_temp[threadIdx.x] * C[INDEX(79)] * exp(2.6026896854443837e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1629
  fwd_rxn_rates[INDEX(1629)] = C[INDEX(48)] * shared_temp[threadIdx.x] * exp(3.0633909220278057e+00 + 2.0 * logT - (3.8555274044727444e+03 / T));
  //rxn 1630
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(56)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(24)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(25)];
  fwd_rxn_rates[INDEX(1630)] = C[INDEX(14)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.9052747784384296e+00 + 2.0 * logT - (3.7458154749421496e+03 / T));
  //rxn 1631
  fwd_rxn_rates[INDEX(1631)] = C[INDEX(34)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.7206395917226240e+00 + 2.0 * logT - (3.1398284105474486e+03 / T));
  //rxn 1632
  fwd_rxn_rates[INDEX(1632)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.4998096703302650e+00 + 2.0 * logT - (4.6744870531071401e+03 / T));
  //rxn 1633
  fwd_rxn_rates[INDEX(1633)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(60)] * exp(4.1067670822206574e+00 + 2.0 * logT - (4.6744870531071401e+03 / T));
  //rxn 1634
  fwd_rxn_rates[INDEX(1634)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(59)] * exp(4.1067670822206574e+00 + 2.0 * logT - (6.4037809589630588e+03 / T));
  //rxn 1635
  shared_temp[threadIdx.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1635)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(63)] * exp(4.7999142627806028e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1636
  fwd_rxn_rates[INDEX(1636)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(64)] * exp(4.7999142627806028e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1637
  fwd_rxn_rates[INDEX(1637)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(62)] * exp(5.4912078746912254e+00 + 2.0 * logT - (7.9559818355537409e+03 / T));
  //rxn 1638
  fwd_rxn_rates[INDEX(1638)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(102)] * exp(4.7999142627806028e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1639
  fwd_rxn_rates[INDEX(1639)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(99)] * exp(4.1067670822206574e+00 + 2.0 * logT - (4.6200839859502157e+03 / T));
  //rxn 1640
  fwd_rxn_rates[INDEX(1640)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(98)] * exp(5.0304379213924353e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1641
  fwd_rxn_rates[INDEX(1641)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(94)] * exp(4.7999142627806028e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1642
  fwd_rxn_rates[INDEX(1642)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(104)] * exp(5.2152074598250300e+00 + 2.0 * logT - (3.8555274044727444e+03 / T));
  //rxn 1643
  fwd_rxn_rates[INDEX(1643)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(108)] * exp(1.2022750588402635e+01 + 1.0 * logT - (1.9448027167025034e+03 / T));
  //rxn 1644
  fwd_rxn_rates[INDEX(1644)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(5.0304379213924353e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1645
  fwd_rxn_rates[INDEX(1645)] = C[INDEX(5)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.2662480039144786e+00 + 2.0 * logT - (2.3609552324497336e+03 / T));
  //rxn 1646
  fwd_rxn_rates[INDEX(1646)] = C[INDEX(4)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(8.9424609268205817e+00 + 2.0 * logT - (2.4291741927598665e+04 / T));
  //rxn 1647
  fwd_rxn_rates[INDEX(1647)] = C[INDEX(7)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(1.5725053062848712e+01 + 1.0 * logT - (7.7533554455675505e+02 / T));
  //rxn 1648
  fwd_rxn_rates[INDEX(1648)] = C[INDEX(9)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(8.1886891244442008e+00 + 2.0 * logT - (9.6570501560167086e+03 / T));
  //rxn 1649
  fwd_rxn_rates[INDEX(1649)] = C[INDEX(22)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.2606154993640253e+00 + 2.0 * logT - (1.7977518954593588e+03 / T));
  //rxn 1650
  fwd_rxn_rates[INDEX(1650)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.8469392014254655e+00 + 2.0 * logT - (6.5649219280222596e+03 / T));
  //rxn 1651
  fwd_rxn_rates[INDEX(1651)] = shared_temp[threadIdx.x + 3 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.3403593037277517e+00 + 2.0 * logT - (7.6891697900697645e+03 / T));
  //rxn 1652
  fwd_rxn_rates[INDEX(1652)] = C[INDEX(45)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.4093521752147034e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1653
  fwd_rxn_rates[INDEX(1653)] = C[INDEX(46)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.6395716127054234e+00 + 2.0 * logT - (7.2616092867871757e+03 / T));
  //rxn 1654
  fwd_rxn_rates[INDEX(1654)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(79)] * exp(4.7999142627806028e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1655
  fwd_rxn_rates[INDEX(1655)] = C[INDEX(48)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.2606154993640253e+00 + 2.0 * logT - (3.8555274044727444e+03 / T));
  //rxn 1656
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(57)];
  fwd_rxn_rates[INDEX(1656)] = C[INDEX(14)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(3.8066624897703196e+00 + 2.0 * logT - (3.7458154749421496e+03 / T));
  //rxn 1657
  fwd_rxn_rates[INDEX(1657)] = C[INDEX(34)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.6220273030545140e+00 + 2.0 * logT - (3.1398284105474486e+03 / T));
  //rxn 1658
  fwd_rxn_rates[INDEX(1658)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(3.4011973816621555e+00 + 2.0 * logT - (4.6744870531071401e+03 / T));
  //rxn 1659
  fwd_rxn_rates[INDEX(1659)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(60)] * exp(3.0081547935525483e+00 + 2.0 * logT - (4.6744870531071401e+03 / T));
  //rxn 1660
  fwd_rxn_rates[INDEX(1660)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(59)] * exp(3.0081547935525483e+00 + 2.0 * logT - (6.4037809589630588e+03 / T));
  //rxn 1661
  fwd_rxn_rates[INDEX(1661)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(63)] * exp(3.7013019741124933e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1662
  fwd_rxn_rates[INDEX(1662)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(64)] * exp(3.7013019741124933e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1663
  fwd_rxn_rates[INDEX(1663)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(62)] * exp(4.3925955860231154e+00 + 2.0 * logT - (7.9559818355537409e+03 / T));
  //rxn 1664
  fwd_rxn_rates[INDEX(1664)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(102)] * exp(3.7013019741124933e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1665
  fwd_rxn_rates[INDEX(1665)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(99)] * exp(3.0081547935525483e+00 + 2.0 * logT - (4.6200839859502157e+03 / T));
  //rxn 1666
  fwd_rxn_rates[INDEX(1666)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(98)] * exp(3.9318256327243257e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1667
  fwd_rxn_rates[INDEX(1667)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(94)] * exp(3.7013019741124933e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1668
  fwd_rxn_rates[INDEX(1668)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(104)] * exp(4.1165951711569209e+00 + 2.0 * logT - (3.8555274044727444e+03 / T));
  //rxn 1669
  fwd_rxn_rates[INDEX(1669)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(108)] * exp(1.0924138299734526e+01 + 1.0 * logT - (1.9448027167025034e+03 / T));
  //rxn 1670
  fwd_rxn_rates[INDEX(1670)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(3.9318256327243257e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1671
  fwd_rxn_rates[INDEX(1671)] = C[INDEX(5)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(8.1676357152463694e+00 + 2.0 * logT - (2.3609552324497336e+03 / T));
  //rxn 1672
  fwd_rxn_rates[INDEX(1672)] = C[INDEX(4)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(7.8438486381524717e+00 + 2.0 * logT - (2.4291741927598665e+04 / T));
  //rxn 1673
  fwd_rxn_rates[INDEX(1673)] = C[INDEX(7)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(1.4626440774180603e+01 + 1.0 * logT - (7.7533554455675505e+02 / T));
  //rxn 1674
  fwd_rxn_rates[INDEX(1674)] = C[INDEX(9)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(7.0900768357760917e+00 + 2.0 * logT - (9.6570501560167086e+03 / T));
  //rxn 1675
  fwd_rxn_rates[INDEX(1675)] = C[INDEX(22)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.1620032106959153e+00 + 2.0 * logT - (1.7977518954593588e+03 / T));
  //rxn 1676
  fwd_rxn_rates[INDEX(1676)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(3.7483269127573564e+00 + 2.0 * logT - (6.5649219280222596e+03 / T));
  //rxn 1677
  fwd_rxn_rates[INDEX(1677)] = shared_temp[threadIdx.x + 3 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.2417470150596426e+00 + 2.0 * logT - (7.6891697900697645e+03 / T));
  //rxn 1678
  fwd_rxn_rates[INDEX(1678)] = C[INDEX(45)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.3107398865465942e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1679
  fwd_rxn_rates[INDEX(1679)] = C[INDEX(46)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(3.5409593240373143e+00 + 2.0 * logT - (7.2616092867871757e+03 / T));
  //rxn 1680
  fwd_rxn_rates[INDEX(1680)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(79)] * exp(3.7013019741124933e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1681
  fwd_rxn_rates[INDEX(1681)] = C[INDEX(48)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.1620032106959153e+00 + 2.0 * logT - (3.8555274044727444e+03 / T));
  //rxn 1682
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(19)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(20)];
  fwd_rxn_rates[INDEX(1682)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(63)] * exp(3.9889840465642745e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1683
  fwd_rxn_rates[INDEX(1683)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(64)] * exp(3.9889840465642745e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1684
  fwd_rxn_rates[INDEX(1684)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(102)] * exp(3.9889840465642745e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1685
  fwd_rxn_rates[INDEX(1685)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(98)] * exp(4.2195077051761070e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1686
  fwd_rxn_rates[INDEX(1686)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(104)] * exp(4.4042772436087017e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1687
  fwd_rxn_rates[INDEX(1687)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(108)] * exp(1.1211820372186306e+01 + 1.0 * logT - (1.0064391297183255e+03 / T));
  //rxn 1688
  fwd_rxn_rates[INDEX(1688)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(110)] * exp(4.2195077051761070e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1689
  fwd_rxn_rates[INDEX(1689)] = C[INDEX(5)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(8.4553177876981493e+00 + 2.0 * logT - (1.2580489121479068e+03 / T));
  //rxn 1690
  fwd_rxn_rates[INDEX(1690)] = C[INDEX(7)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(1.4914122846632385e+01 + 1.0 * logT - (2.5160978242958137e+01 / T));
  //rxn 1691
  fwd_rxn_rates[INDEX(1691)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(22)] * exp(4.4496852831476961e+00 + 2.0 * logT - (7.5482934728874409e+02 / T));
  //rxn 1692
  fwd_rxn_rates[INDEX(1692)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(46)] * exp(3.8286413964890951e+00 + 2.0 * logT - (5.7870249958803715e+03 / T));
  //rxn 1693
  fwd_rxn_rates[INDEX(1693)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(48)] * exp(4.4496852831476961e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1694
  shared_temp[threadIdx.x] = C[INDEX(7)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(31)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(32)];
  fwd_rxn_rates[INDEX(1694)] = C[INDEX(3)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.3926619287701367e+00 + 2.0 * logT - (2.0128782594366510e+03 / T));
  //rxn 1695
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(44)];
  fwd_rxn_rates[INDEX(1695)] = C[INDEX(14)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.7874917427820458e+00 + 2.0 * logT - (2.5160978242958136e+03 / T));
  //rxn 1696
  fwd_rxn_rates[INDEX(1696)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(34)] * exp(5.6028565560662402e+00 + 2.0 * logT - (1.9625563029507348e+03 / T));
  //rxn 1697
  fwd_rxn_rates[INDEX(1697)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3820266346738812e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1698
  fwd_rxn_rates[INDEX(1698)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(60)] * exp(3.9889840465642745e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1699
  fwd_rxn_rates[INDEX(1699)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(59)] * exp(3.9889840465642745e+00 + 2.0 * logT - (4.9818736921057116e+03 / T));
  //rxn 1700
  fwd_rxn_rates[INDEX(1700)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(63)] * exp(4.6821312271242199e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1701
  fwd_rxn_rates[INDEX(1701)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(64)] * exp(4.6821312271242199e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1702
  fwd_rxn_rates[INDEX(1702)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(62)] * exp(5.3734248390348425e+00 + 2.0 * logT - (6.4412104301972831e+03 / T));
  //rxn 1703
  fwd_rxn_rates[INDEX(1703)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(102)] * exp(4.6821312271242199e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1704
  fwd_rxn_rates[INDEX(1704)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(99)] * exp(3.9889840465642745e+00 + 2.0 * logT - (3.3212491280704739e+03 / T));
  //rxn 1705
  fwd_rxn_rates[INDEX(1705)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(98)] * exp(4.9126548857360524e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1706
  fwd_rxn_rates[INDEX(1706)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(94)] * exp(4.6821312271242199e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1707
  fwd_rxn_rates[INDEX(1707)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(104)] * exp(5.0974244241686471e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1708
  fwd_rxn_rates[INDEX(1708)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(108)] * exp(1.1904967552746252e+01 + 1.0 * logT - (1.0064391297183255e+03 / T));
  //rxn 1709
  fwd_rxn_rates[INDEX(1709)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(4.9126548857360524e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1710
  fwd_rxn_rates[INDEX(1710)] = C[INDEX(5)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.1484649682580947e+00 + 2.0 * logT - (1.2580489121479068e+03 / T));
  //rxn 1711
  fwd_rxn_rates[INDEX(1711)] = C[INDEX(4)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(8.8246778911641979e+00 + 2.0 * logT - (2.2141660853803161e+04 / T));
  //rxn 1712
  fwd_rxn_rates[INDEX(1712)] = shared_temp[threadIdx.x] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(1.5607270027192330e+01 + 1.0 * logT - (2.5160978242958137e+01 / T));
  //rxn 1713
  fwd_rxn_rates[INDEX(1713)] = C[INDEX(9)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(8.0709060887878188e+00 + 2.0 * logT - (8.0515130377466039e+03 / T));
  //rxn 1714
  fwd_rxn_rates[INDEX(1714)] = C[INDEX(22)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.1428324637076415e+00 + 2.0 * logT - (7.5482934728874409e+02 / T));
  //rxn 1715
  fwd_rxn_rates[INDEX(1715)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.7291561657690826e+00 + 2.0 * logT - (5.1328395615634599e+03 / T));
  //rxn 1716
  fwd_rxn_rates[INDEX(1716)] = C[INDEX(25)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.2225762680713688e+00 + 2.0 * logT - (6.1896006477677020e+03 / T));
  //rxn 1717
  fwd_rxn_rates[INDEX(1717)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(45)] * exp(6.2915691395583204e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1718
  fwd_rxn_rates[INDEX(1718)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(46)] * exp(4.5217885770490405e+00 + 2.0 * logT - (5.7870249958803715e+03 / T));
  //rxn 1719
  fwd_rxn_rates[INDEX(1719)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(79)] * exp(4.6821312271242199e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1720
  fwd_rxn_rates[INDEX(1720)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(48)] * exp(5.1428324637076415e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1721
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(10)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(34)];
  shared_temp[threadIdx.x] = C[INDEX(84)];
  fwd_rxn_rates[INDEX(1721)] = C[INDEX(3)] * shared_temp[threadIdx.x] * exp(9.9804485936722571e+00 + 2.0 * logT - (1.2950909043171898e+03 / T));
  //rxn 1722
  fwd_rxn_rates[INDEX(1722)] = C[INDEX(14)] * shared_temp[threadIdx.x] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.7694810203055713e+03 / T));
  //rxn 1723
  fwd_rxn_rates[INDEX(1723)] = shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x] * exp(6.1906432209683597e+00 + 2.0 * logT - (1.2477832330247800e+03 / T));
  //rxn 1724
  fwd_rxn_rates[INDEX(1724)] = C[INDEX(30)] * shared_temp[threadIdx.x] * exp(4.9698132995760007e+00 + 2.0 * logT - (2.5805149607934354e+03 / T));
  //rxn 1725
  fwd_rxn_rates[INDEX(1725)] = C[INDEX(60)] * shared_temp[threadIdx.x] * exp(4.5767707114663931e+00 + 2.0 * logT - (2.5805149607934354e+03 / T));
  //rxn 1726
  fwd_rxn_rates[INDEX(1726)] = C[INDEX(59)] * shared_temp[threadIdx.x] * exp(4.5767707114663931e+00 + 2.0 * logT - (4.1185753676115746e+03 / T));
  //rxn 1727
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1727)] = C[INDEX(63)] * shared_temp[threadIdx.x] * exp(5.2699178920263385e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1728
  fwd_rxn_rates[INDEX(1728)] = C[INDEX(64)] * shared_temp[threadIdx.x] * exp(5.2699178920263385e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1729
  fwd_rxn_rates[INDEX(1729)] = C[INDEX(62)] * shared_temp[threadIdx.x] * exp(5.9612115039369611e+00 + 2.0 * logT - (5.5215313856563262e+03 / T));
  //rxn 1730
  fwd_rxn_rates[INDEX(1730)] = shared_temp[threadIdx.x] * C[INDEX(102)] * exp(5.2699178920263385e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1731
  fwd_rxn_rates[INDEX(1731)] = shared_temp[threadIdx.x] * C[INDEX(99)] * exp(4.5767707114663931e+00 + 2.0 * logT - (2.5326688445666259e+03 / T));
  //rxn 1732
  fwd_rxn_rates[INDEX(1732)] = shared_temp[threadIdx.x] * C[INDEX(98)] * exp(5.5004415506381710e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1733
  fwd_rxn_rates[INDEX(1733)] = shared_temp[threadIdx.x] * C[INDEX(94)] * exp(5.2699178920263385e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1734
  fwd_rxn_rates[INDEX(1734)] = shared_temp[threadIdx.x] * C[INDEX(104)] * exp(5.6852110890707657e+00 + 2.0 * logT - (1.8646197112378445e+03 / T));
  //rxn 1735
  fwd_rxn_rates[INDEX(1735)] = shared_temp[threadIdx.x] * C[INDEX(108)] * exp(1.2492754217648372e+01 + 1.0 * logT - (4.3671909936302438e+02 / T));
  //rxn 1736
  fwd_rxn_rates[INDEX(1736)] = shared_temp[threadIdx.x] * C[INDEX(110)] * exp(5.5004415506381710e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1737
  fwd_rxn_rates[INDEX(1737)] = C[INDEX(5)] * shared_temp[threadIdx.x] * exp(9.7362516331602151e+00 + 2.0 * logT - (5.8842470158111621e+02 / T));
  //rxn 1738
  fwd_rxn_rates[INDEX(1738)] = C[INDEX(4)] * shared_temp[threadIdx.x] * exp(9.4124645560663165e+00 + 2.0 * logT - (2.0836253948406356e+04 / T));
  //rxn 1739
  fwd_rxn_rates[INDEX(1739)] = C[INDEX(9)] * shared_temp[threadIdx.x] * exp(8.6586927536899374e+00 + 2.0 * logT - (7.0767213864622709e+03 / T));
  //rxn 1740
  fwd_rxn_rates[INDEX(1740)] = C[INDEX(22)] * shared_temp[threadIdx.x] * exp(5.7306191286097610e+00 + 2.0 * logT - (1.2162816882645963e+02 / T));
  //rxn 1741
  fwd_rxn_rates[INDEX(1741)] = C[INDEX(23)] * shared_temp[threadIdx.x] * exp(5.3169428306712012e+00 + 2.0 * logT - (4.2633617008128531e+03 / T));
  //rxn 1742
  fwd_rxn_rates[INDEX(1742)] = C[INDEX(25)] * shared_temp[threadIdx.x] * exp(6.8103629329734874e+00 + 2.0 * logT - (5.2791506500462619e+03 / T));
  //rxn 1743
  fwd_rxn_rates[INDEX(1743)] = C[INDEX(45)] * shared_temp[threadIdx.x] * exp(6.8793558044604390e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1744
  fwd_rxn_rates[INDEX(1744)] = C[INDEX(46)] * shared_temp[threadIdx.x] * exp(5.1095752419511591e+00 + 2.0 * logT - (4.8917420358437867e+03 / T));
  //rxn 1745
  fwd_rxn_rates[INDEX(1745)] = C[INDEX(79)] * shared_temp[threadIdx.x] * exp(5.2699178920263385e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1746
  fwd_rxn_rates[INDEX(1746)] = C[INDEX(48)] * shared_temp[threadIdx.x] * exp(5.7306191286097610e+00 + 2.0 * logT - (1.8646197112378445e+03 / T));
  //rxn 1747
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(66)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(25)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(33)];
  fwd_rxn_rates[INDEX(1747)] = C[INDEX(3)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.4118326757584114e+00 + 2.0 * logT - (2.0128782594366510e+03 / T));
  //rxn 1748
  shared_temp[threadIdx.x] = C[INDEX(86)];
  fwd_rxn_rates[INDEX(1748)] = C[INDEX(14)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.8066624897703196e+00 + 2.0 * logT - (2.5160978242958136e+03 / T));
  //rxn 1749
  fwd_rxn_rates[INDEX(1749)] = C[INDEX(34)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.6220273030545140e+00 + 2.0 * logT - (1.9625563029507348e+03 / T));
}

