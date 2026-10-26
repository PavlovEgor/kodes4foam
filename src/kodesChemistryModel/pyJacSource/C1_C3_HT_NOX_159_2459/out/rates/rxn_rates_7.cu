#include "rates.cuh"
__device__ void eval_rxn_rates_7 (const double T, const double pres, const double * __restrict__ C, double * __restrict__ fwd_rxn_rates, double * __restrict__ rev_rxn_rates) {
  extern volatile __shared__ double shared_temp[];
  register double logT = log(T);

  register double kf;
  register double Kc;

  //rxn 1750
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(66)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(25)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(33)];
  shared_temp[threadIdx.x] = C[INDEX(86)];
  fwd_rxn_rates[INDEX(1750)] = C[INDEX(30)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.4011973816621555e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1751
  fwd_rxn_rates[INDEX(1751)] = C[INDEX(60)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.0081547935525483e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1752
  fwd_rxn_rates[INDEX(1752)] = C[INDEX(59)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.0081547935525483e+00 + 2.0 * logT - (4.9818736921057116e+03 / T));
  //rxn 1753
  fwd_rxn_rates[INDEX(1753)] = C[INDEX(63)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.7013019741124933e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1754
  fwd_rxn_rates[INDEX(1754)] = C[INDEX(64)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.7013019741124933e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1755
  fwd_rxn_rates[INDEX(1755)] = C[INDEX(62)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.3925955860231154e+00 + 2.0 * logT - (6.4412104301972831e+03 / T));
  //rxn 1756
  fwd_rxn_rates[INDEX(1756)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(102)] * exp(3.7013019741124933e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1757
  fwd_rxn_rates[INDEX(1757)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(99)] * exp(3.0081547935525483e+00 + 2.0 * logT - (3.3212491280704739e+03 / T));
  //rxn 1758
  fwd_rxn_rates[INDEX(1758)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(98)] * exp(3.9318256327243257e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1759
  fwd_rxn_rates[INDEX(1759)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(94)] * exp(3.7013019741124933e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1760
  fwd_rxn_rates[INDEX(1760)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(104)] * exp(4.1165951711569209e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1761
  fwd_rxn_rates[INDEX(1761)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(108)] * exp(1.0924138299734526e+01 + 1.0 * logT - (1.0064391297183255e+03 / T));
  //rxn 1762
  fwd_rxn_rates[INDEX(1762)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(110)] * exp(3.9318256327243257e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1763
  fwd_rxn_rates[INDEX(1763)] = C[INDEX(5)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.1676357152463694e+00 + 2.0 * logT - (1.2580489121479068e+03 / T));
  //rxn 1764
  fwd_rxn_rates[INDEX(1764)] = C[INDEX(4)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.8438486381524717e+00 + 2.0 * logT - (2.2141660853803161e+04 / T));
  //rxn 1765
  fwd_rxn_rates[INDEX(1765)] = C[INDEX(7)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(1.4626440774180603e+01 + 1.0 * logT - (2.5160978242958137e+01 / T));
  //rxn 1766
  fwd_rxn_rates[INDEX(1766)] = C[INDEX(9)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.0900768357760917e+00 + 2.0 * logT - (8.0515130377466039e+03 / T));
  //rxn 1767
  fwd_rxn_rates[INDEX(1767)] = C[INDEX(22)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.1620032106959153e+00 + 2.0 * logT - (7.5482934728874409e+02 / T));
  //rxn 1768
  fwd_rxn_rates[INDEX(1768)] = C[INDEX(23)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.7483269127573564e+00 + 2.0 * logT - (5.1328395615634599e+03 / T));
  //rxn 1769
  fwd_rxn_rates[INDEX(1769)] = shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.2417470150596426e+00 + 2.0 * logT - (6.1896006477677020e+03 / T));
  //rxn 1770
  fwd_rxn_rates[INDEX(1770)] = C[INDEX(45)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.3107398865465942e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1771
  fwd_rxn_rates[INDEX(1771)] = C[INDEX(46)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.5409593240373143e+00 + 2.0 * logT - (5.7870249958803715e+03 / T));
  //rxn 1772
  fwd_rxn_rates[INDEX(1772)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(79)] * exp(3.7013019741124933e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1773
  fwd_rxn_rates[INDEX(1773)] = C[INDEX(48)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.1620032106959153e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1774
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(3)];
  shared_temp[threadIdx.x] = C[INDEX(84)];
  fwd_rxn_rates[INDEX(1774)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.4118326757584114e+00 + 2.0 * logT - (2.0128782594366510e+03 / T));
  //rxn 1775
  fwd_rxn_rates[INDEX(1775)] = C[INDEX(14)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.8066624897703196e+00 + 2.0 * logT - (2.5160978242958136e+03 / T));
  //rxn 1776
  fwd_rxn_rates[INDEX(1776)] = C[INDEX(34)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.6220273030545140e+00 + 2.0 * logT - (1.9625563029507348e+03 / T));
  //rxn 1777
  fwd_rxn_rates[INDEX(1777)] = C[INDEX(30)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.4011973816621555e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1778
  fwd_rxn_rates[INDEX(1778)] = C[INDEX(60)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.0081547935525483e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1779
  fwd_rxn_rates[INDEX(1779)] = C[INDEX(59)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.0081547935525483e+00 + 2.0 * logT - (4.9818736921057116e+03 / T));
  //rxn 1780
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1780)] = C[INDEX(63)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.7013019741124933e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1781
  fwd_rxn_rates[INDEX(1781)] = C[INDEX(64)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.7013019741124933e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1782
  fwd_rxn_rates[INDEX(1782)] = C[INDEX(62)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.3925955860231154e+00 + 2.0 * logT - (6.4412104301972831e+03 / T));
  //rxn 1783
  fwd_rxn_rates[INDEX(1783)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(102)] * exp(3.7013019741124933e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1784
  fwd_rxn_rates[INDEX(1784)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(99)] * exp(3.0081547935525483e+00 + 2.0 * logT - (3.3212491280704739e+03 / T));
  //rxn 1785
  fwd_rxn_rates[INDEX(1785)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(98)] * exp(3.9318256327243257e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1786
  fwd_rxn_rates[INDEX(1786)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(94)] * exp(3.7013019741124933e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1787
  fwd_rxn_rates[INDEX(1787)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(104)] * exp(4.1165951711569209e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1788
  fwd_rxn_rates[INDEX(1788)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(108)] * exp(1.0924138299734526e+01 + 1.0 * logT - (1.0064391297183255e+03 / T));
  //rxn 1789
  fwd_rxn_rates[INDEX(1789)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(110)] * exp(3.9318256327243257e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1790
  fwd_rxn_rates[INDEX(1790)] = C[INDEX(5)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.1676357152463694e+00 + 2.0 * logT - (1.2580489121479068e+03 / T));
  //rxn 1791
  fwd_rxn_rates[INDEX(1791)] = C[INDEX(4)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.8438486381524717e+00 + 2.0 * logT - (2.2141660853803161e+04 / T));
  //rxn 1792
  fwd_rxn_rates[INDEX(1792)] = C[INDEX(7)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(1.4626440774180603e+01 + 1.0 * logT - (2.5160978242958137e+01 / T));
  //rxn 1793
  fwd_rxn_rates[INDEX(1793)] = C[INDEX(9)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.0900768357760917e+00 + 2.0 * logT - (8.0515130377466039e+03 / T));
  //rxn 1794
  fwd_rxn_rates[INDEX(1794)] = C[INDEX(22)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.1620032106959153e+00 + 2.0 * logT - (7.5482934728874409e+02 / T));
  //rxn 1795
  fwd_rxn_rates[INDEX(1795)] = C[INDEX(23)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.7483269127573564e+00 + 2.0 * logT - (5.1328395615634599e+03 / T));
  //rxn 1796
  fwd_rxn_rates[INDEX(1796)] = C[INDEX(25)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.2417470150596426e+00 + 2.0 * logT - (6.1896006477677020e+03 / T));
  //rxn 1797
  fwd_rxn_rates[INDEX(1797)] = C[INDEX(45)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.3107398865465942e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1798
  fwd_rxn_rates[INDEX(1798)] = C[INDEX(46)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.5409593240373143e+00 + 2.0 * logT - (5.7870249958803715e+03 / T));
  //rxn 1799
  fwd_rxn_rates[INDEX(1799)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(79)] * exp(3.7013019741124933e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1800
  fwd_rxn_rates[INDEX(1800)] = C[INDEX(48)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.1620032106959153e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1801
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(103)];
  shared_temp[threadIdx.x] = C[INDEX(104)];
  fwd_rxn_rates[INDEX(1801)] = C[INDEX(34)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.8556195245616083e+00 + 2.0 * logT - (4.3899969043096607e+03 / T));
  //rxn 1802
  fwd_rxn_rates[INDEX(1802)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.6347896031692493e+00 + 2.0 * logT - (6.3655664652076539e+03 / T));
  //rxn 1803
  fwd_rxn_rates[INDEX(1803)] = C[INDEX(60)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.2417470150596426e+00 + 2.0 * logT - (6.3655664652076539e+03 / T));
  //rxn 1804
  fwd_rxn_rates[INDEX(1804)] = C[INDEX(59)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.2417470150596426e+00 + 2.0 * logT - (8.2648479400729975e+03 / T));
  //rxn 1805
  fwd_rxn_rates[INDEX(1805)] = C[INDEX(63)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.9348941956195880e+00 + 2.0 * logT - (5.1944235719109238e+03 / T));
  //rxn 1806
  fwd_rxn_rates[INDEX(1806)] = C[INDEX(64)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.9348941956195880e+00 + 2.0 * logT - (5.1944235719109238e+03 / T));
  //rxn 1807
  fwd_rxn_rates[INDEX(1807)] = C[INDEX(62)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.6261878075302105e+00 + 2.0 * logT - (1.0119936672752410e+04 / T));
  //rxn 1808
  fwd_rxn_rates[INDEX(1808)] = C[INDEX(102)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.9348941956195880e+00 + 2.0 * logT - (4.7627718935660323e+03 / T));
  //rxn 1809
  fwd_rxn_rates[INDEX(1809)] = C[INDEX(99)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.2417470150596426e+00 + 2.0 * logT - (6.4755652298902187e+03 / T));
  //rxn 1810
  fwd_rxn_rates[INDEX(1810)] = C[INDEX(98)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.1654178542314204e+00 + 2.0 * logT - (1.0637477866427464e+04 / T));
  //rxn 1811
  fwd_rxn_rates[INDEX(1811)] = C[INDEX(94)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.9348941956195880e+00 + 2.0 * logT - (1.0637477866427464e+04 / T));
  //rxn 1812
  fwd_rxn_rates[INDEX(1812)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(108)] * exp(1.3157730521241620e+01 + 1.0 * logT - (2.8536675727946372e+03 / T));
  //rxn 1813
  fwd_rxn_rates[INDEX(1813)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(6.1654178542314204e+00 + 2.0 * logT - (1.0809503474674568e+04 / T));
  //rxn 1814
  fwd_rxn_rates[INDEX(1814)] = C[INDEX(5)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(1.0401227936753463e+01 + 2.0 * logT - (3.9365407222194212e+03 / T));
  //rxn 1815
  fwd_rxn_rates[INDEX(1815)] = C[INDEX(9)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.3236690572831851e+00 + 2.0 * logT - (1.2245536114717514e+04 / T));
  //rxn 1816
  fwd_rxn_rates[INDEX(1816)] = C[INDEX(22)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.3955954322030095e+00 + 2.0 * logT - (3.1174150111286217e+03 / T));
  //rxn 1817
  fwd_rxn_rates[INDEX(1817)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.9819191342644507e+00 + 2.0 * logT - (8.4405218901653316e+03 / T));
  //rxn 1818
  fwd_rxn_rates[INDEX(1818)] = C[INDEX(25)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(7.4753392365667368e+00 + 2.0 * logT - (9.9347619372755362e+03 / T));
  //rxn 1819
  fwd_rxn_rates[INDEX(1819)] = C[INDEX(45)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(7.5443321080536885e+00 + 2.0 * logT - (1.0912859741100992e+04 / T));
  //rxn 1820
  fwd_rxn_rates[INDEX(1820)] = C[INDEX(46)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.7745515455444085e+00 + 2.0 * logT - (9.4715131024531365e+03 / T));
  //rxn 1821
  fwd_rxn_rates[INDEX(1821)] = C[INDEX(79)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.9348941956195880e+00 + 2.0 * logT - (1.0377851796329676e+04 / T));
  //rxn 1822
  fwd_rxn_rates[INDEX(1822)] = C[INDEX(48)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.3955954322030095e+00 + 2.0 * logT - (5.1935680986506641e+03 / T));
  //rxn 1823
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(101)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(102)];
  fwd_rxn_rates[INDEX(1823)] = C[INDEX(3)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(1.1184421397998193e+01 + 2.0 * logT - (4.7104521554076246e+03 / T));
  //rxn 1824
  fwd_rxn_rates[INDEX(1824)] = C[INDEX(14)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.5792512120101012e+00 + 2.0 * logT - (5.1423554435349470e+03 / T));
  //rxn 1825
  fwd_rxn_rates[INDEX(1825)] = C[INDEX(30)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.1737861039019366e+00 + 2.0 * logT - (6.4728277154573843e+03 / T));
  //rxn 1826
  fwd_rxn_rates[INDEX(1826)] = C[INDEX(60)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.7807435157923290e+00 + 2.0 * logT - (6.4728277154573843e+03 / T));
  //rxn 1827
  fwd_rxn_rates[INDEX(1827)] = C[INDEX(59)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.7807435157923290e+00 + 2.0 * logT - (8.3848507097049624e+03 / T));
  //rxn 1828
  fwd_rxn_rates[INDEX(1828)] = C[INDEX(63)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.4738906963522744e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1829
  fwd_rxn_rates[INDEX(1829)] = C[INDEX(64)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.4738906963522744e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1830
  fwd_rxn_rates[INDEX(1830)] = C[INDEX(99)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.7807435157923290e+00 + 2.0 * logT - (6.6147255683563708e+03 / T));
  //rxn 1831
  fwd_rxn_rates[INDEX(1831)] = C[INDEX(98)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.7044143549641069e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1832
  fwd_rxn_rates[INDEX(1832)] = C[INDEX(94)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.4738906963522744e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1833
  fwd_rxn_rates[INDEX(1833)] = shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x] * exp(6.8891838933967016e+00 + 2.0 * logT - (5.2659914584251947e+03 / T));
  //rxn 1834
  fwd_rxn_rates[INDEX(1834)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(108)] * exp(1.3696727021974308e+01 + 1.0 * logT - (2.8721961171727517e+03 / T));
  //rxn 1835
  fwd_rxn_rates[INDEX(1835)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(110)] * exp(6.7044143549641069e+00 + 2.0 * logT - (1.0975581027665039e+04 / T));
  //rxn 1836
  fwd_rxn_rates[INDEX(1836)] = C[INDEX(22)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.9345919329356969e+00 + 2.0 * logT - (3.1968129940721005e+03 / T));
  //rxn 1837
  fwd_rxn_rates[INDEX(1837)] = C[INDEX(23)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.5209156349971380e+00 + 2.0 * logT - (8.5616166462530400e+03 / T));
  //rxn 1838
  fwd_rxn_rates[INDEX(1838)] = C[INDEX(25)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(8.0143357372994242e+00 + 2.0 * logT - (1.0115065507364574e+04 / T));
  //rxn 1839
  fwd_rxn_rates[INDEX(1839)] = C[INDEX(45)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(8.0833286087863758e+00 + 2.0 * logT - (1.1098572921512266e+04 / T));
  //rxn 1840
  fwd_rxn_rates[INDEX(1840)] = C[INDEX(46)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.3135480462770950e+00 + 2.0 * logT - (9.6491395444571226e+03 / T));
  //rxn 1841
  fwd_rxn_rates[INDEX(1841)] = C[INDEX(48)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.9345919329356969e+00 + 2.0 * logT - (5.2442875985928185e+03 / T));
  //rxn 1842
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(96)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(98)];
  fwd_rxn_rates[INDEX(1842)] = C[INDEX(3)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.7981270368783022e+00 + 2.0 * logT - (2.2090735033839414e+03 / T));
  //rxn 1843
  fwd_rxn_rates[INDEX(1843)] = C[INDEX(14)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.1929568508902104e+00 + 2.0 * logT - (2.8375746111104413e+03 / T));
  //rxn 1844
  fwd_rxn_rates[INDEX(1844)] = C[INDEX(34)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.0083216641744048e+00 + 2.0 * logT - (2.2982188492987420e+03 / T));
  //rxn 1845
  fwd_rxn_rates[INDEX(1845)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.7874917427820458e+00 + 2.0 * logT - (3.5039429910925933e+03 / T));
  //rxn 1846
  fwd_rxn_rates[INDEX(1846)] = C[INDEX(60)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.5039429910925933e+03 / T));
  //rxn 1847
  fwd_rxn_rates[INDEX(1847)] = C[INDEX(59)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (5.1142455986419136e+03 / T));
  //rxn 1848
  shared_temp[threadIdx.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1848)] = C[INDEX(63)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.0875963352323836e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1849
  fwd_rxn_rates[INDEX(1849)] = C[INDEX(64)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.0875963352323836e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1850
  fwd_rxn_rates[INDEX(1850)] = C[INDEX(62)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.7788899471430062e+00 + 2.0 * logT - (6.4412104301972831e+03 / T));
  //rxn 1851
  fwd_rxn_rates[INDEX(1851)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(102)] * exp(5.0875963352323836e+00 + 2.0 * logT - (2.6001505882142396e+03 / T));
  //rxn 1852
  fwd_rxn_rates[INDEX(1852)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(99)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.3212491280704739e+03 / T));
  //rxn 1853
  fwd_rxn_rates[INDEX(1853)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.3181199938442161e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1854
  fwd_rxn_rates[INDEX(1854)] = C[INDEX(94)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.0875963352323836e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1855
  fwd_rxn_rates[INDEX(1855)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(104)] * exp(5.5028895322768108e+00 + 2.0 * logT - (2.9382185240822737e+03 / T));
  //rxn 1856
  fwd_rxn_rates[INDEX(1856)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(108)] * exp(1.2310432660854417e+01 + 1.0 * logT - (1.3421016760663329e+03 / T));
  //rxn 1857
  fwd_rxn_rates[INDEX(1857)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(5.3181199938442161e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1858
  fwd_rxn_rates[INDEX(1858)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.1346212738772472e+00 + 2.0 * logT - (5.2652114680996629e+03 / T));
  //rxn 1859
  fwd_rxn_rates[INDEX(1859)] = C[INDEX(25)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.6280413761795334e+00 + 2.0 * logT - (6.1092314510640454e+03 / T));
  //rxn 1860
  fwd_rxn_rates[INDEX(1860)] = C[INDEX(45)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.6970342476664841e+00 + 2.0 * logT - (6.9647047113246217e+03 / T));
  //rxn 1861
  fwd_rxn_rates[INDEX(1861)] = C[INDEX(46)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.9272536851572051e+00 + 2.0 * logT - (5.7066557991767149e+03 / T));
  //rxn 1862
  fwd_rxn_rates[INDEX(1862)] = C[INDEX(79)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.0875963352323836e+00 + 2.0 * logT - (7.3807364543762860e+03 / T));
  //rxn 1863
  fwd_rxn_rates[INDEX(1863)] = C[INDEX(48)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.5482975718158061e+00 + 2.0 * logT - (2.9524042836156536e+03 / T));
  //rxn 1864
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(99)];
  fwd_rxn_rates[INDEX(1864)] = C[INDEX(3)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.1049798563183568e+00 + 2.0 * logT - (3.0318123309504294e+03 / T));
  //rxn 1865
  fwd_rxn_rates[INDEX(1865)] = C[INDEX(14)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.4998096703302650e+00 + 2.0 * logT - (3.4782385357195872e+03 / T));
  //rxn 1866
  fwd_rxn_rates[INDEX(1866)] = C[INDEX(34)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.3151744836144594e+00 + 2.0 * logT - (2.8604459403332903e+03 / T));
  //rxn 1867
  fwd_rxn_rates[INDEX(1867)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.0943445622221004e+00 + 2.0 * logT - (4.5643071293812263e+03 / T));
  //rxn 1868
  fwd_rxn_rates[INDEX(1868)] = C[INDEX(60)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(3.7013019741124933e+00 + 2.0 * logT - (4.5643071293812263e+03 / T));
  //rxn 1869
  fwd_rxn_rates[INDEX(1869)] = C[INDEX(59)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(3.7013019741124933e+00 + 2.0 * logT - (6.2936010352371450e+03 / T));
  //rxn 1870
  fwd_rxn_rates[INDEX(1870)] = C[INDEX(63)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1871
  fwd_rxn_rates[INDEX(1871)] = C[INDEX(64)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1872
  fwd_rxn_rates[INDEX(1872)] = C[INDEX(62)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.0857427665830608e+00 + 2.0 * logT - (7.9559818355537409e+03 / T));
  //rxn 1873
  fwd_rxn_rates[INDEX(1873)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(102)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.1915493174236735e+03 / T));
  //rxn 1874
  fwd_rxn_rates[INDEX(1874)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.6249728132842707e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1875
  fwd_rxn_rates[INDEX(1875)] = C[INDEX(94)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1876
  fwd_rxn_rates[INDEX(1876)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(104)] * exp(4.8097423517168654e+00 + 2.0 * logT - (3.5879504652501819e+03 / T));
  //rxn 1877
  fwd_rxn_rates[INDEX(1877)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(108)] * exp(1.1617285480294472e+01 + 1.0 * logT - (1.6654202464883449e+03 / T));
  //rxn 1878
  fwd_rxn_rates[INDEX(1878)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(4.6249728132842707e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1879
  fwd_rxn_rates[INDEX(1879)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.4414740933173018e+00 + 2.0 * logT - (6.4547420042963458e+03 / T));
  //rxn 1880
  fwd_rxn_rates[INDEX(1880)] = C[INDEX(25)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.9348941956195880e+00 + 2.0 * logT - (7.7560627668264915e+03 / T));
  //rxn 1881
  fwd_rxn_rates[INDEX(1881)] = C[INDEX(45)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.0038870671065387e+00 + 2.0 * logT - (8.6620290782250377e+03 / T));
  //rxn 1882
  fwd_rxn_rates[INDEX(1882)] = C[INDEX(46)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.2341065045972597e+00 + 2.0 * logT - (7.3285022635439045e+03 / T));
  //rxn 1883
  fwd_rxn_rates[INDEX(1883)] = C[INDEX(79)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (8.3157536312541488e+03 / T));
  //rxn 1884
  fwd_rxn_rates[INDEX(1884)] = C[INDEX(48)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.8551503912558607e+00 + 2.0 * logT - (3.5761449342585856e+03 / T));
  //rxn 1885
  shared_temp[threadIdx.x] = C[INDEX(14)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(77)];
  fwd_rxn_rates[INDEX(1885)] = C[INDEX(3)] * C[INDEX(97)] * exp(9.7981270368783022e+00 + 2.0 * logT - (2.0128782594366510e+03 / T));
  //rxn 1886
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(97)];
  fwd_rxn_rates[INDEX(1886)] = shared_temp[threadIdx.x] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.1929568508902104e+00 + 2.0 * logT - (2.5160978242958136e+03 / T));
  //rxn 1887
  fwd_rxn_rates[INDEX(1887)] = C[INDEX(34)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.0083216641744048e+00 + 2.0 * logT - (1.9625563029507348e+03 / T));
  //rxn 1888
  fwd_rxn_rates[INDEX(1888)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.7874917427820458e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1889
  fwd_rxn_rates[INDEX(1889)] = C[INDEX(60)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1890
  fwd_rxn_rates[INDEX(1890)] = C[INDEX(59)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (4.9818736921057116e+03 / T));
  //rxn 1891
  fwd_rxn_rates[INDEX(1891)] = C[INDEX(63)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.0875963352323836e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1892
  fwd_rxn_rates[INDEX(1892)] = C[INDEX(64)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.0875963352323836e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1893
  fwd_rxn_rates[INDEX(1893)] = C[INDEX(62)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.7788899471430062e+00 + 2.0 * logT - (6.4412104301972831e+03 / T));
  //rxn 1894
  fwd_rxn_rates[INDEX(1894)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(102)] * exp(5.0875963352323836e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1895
  fwd_rxn_rates[INDEX(1895)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(99)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.3212491280704739e+03 / T));
  //rxn 1896
  fwd_rxn_rates[INDEX(1896)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.3181199938442161e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1897
  fwd_rxn_rates[INDEX(1897)] = C[INDEX(94)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.0875963352323836e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1898
  fwd_rxn_rates[INDEX(1898)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(104)] * exp(5.5028895322768108e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1899
  fwd_rxn_rates[INDEX(1899)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(108)] * exp(1.2310432660854417e+01 + 1.0 * logT - (1.0064391297183255e+03 / T));
  //rxn 1900
  fwd_rxn_rates[INDEX(1900)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(5.3181199938442161e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1901
  fwd_rxn_rates[INDEX(1901)] = C[INDEX(5)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.5539300763662602e+00 + 2.0 * logT - (1.2580489121479068e+03 / T));
  //rxn 1902
  fwd_rxn_rates[INDEX(1902)] = C[INDEX(4)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.2301429992723616e+00 + 2.0 * logT - (2.2141660853803161e+04 / T));
  //rxn 1903
  fwd_rxn_rates[INDEX(1903)] = C[INDEX(7)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(1.6012735135300492e+01 + 1.0 * logT - (2.5160978242958137e+01 / T));
  //rxn 1904
  fwd_rxn_rates[INDEX(1904)] = C[INDEX(9)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(8.4763711968959825e+00 + 2.0 * logT - (8.0515130377466039e+03 / T));
  //rxn 1905
  fwd_rxn_rates[INDEX(1905)] = C[INDEX(22)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.5482975718158061e+00 + 2.0 * logT - (7.5482934728874409e+02 / T));
  //rxn 1906
  fwd_rxn_rates[INDEX(1906)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.1346212738772472e+00 + 2.0 * logT - (5.1328395615634599e+03 / T));
  //rxn 1907
  fwd_rxn_rates[INDEX(1907)] = C[INDEX(25)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.6280413761795334e+00 + 2.0 * logT - (6.1896006477677020e+03 / T));
  //rxn 1908
  fwd_rxn_rates[INDEX(1908)] = C[INDEX(45)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.6970342476664841e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1909
  fwd_rxn_rates[INDEX(1909)] = C[INDEX(46)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.9272536851572051e+00 + 2.0 * logT - (5.7870249958803715e+03 / T));
  //rxn 1910
  fwd_rxn_rates[INDEX(1910)] = C[INDEX(79)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.0875963352323836e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1911
  fwd_rxn_rates[INDEX(1911)] = C[INDEX(48)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.5482975718158061e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1912
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(93)];
  shared_temp[threadIdx.x] = C[INDEX(94)];
  fwd_rxn_rates[INDEX(1912)] = C[INDEX(3)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(9.7981270368783022e+00 + 2.0 * logT - (2.2090735033839414e+03 / T));
  //rxn 1913
  fwd_rxn_rates[INDEX(1913)] = C[INDEX(14)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.1929568508902104e+00 + 2.0 * logT - (2.8375746111104413e+03 / T));
  //rxn 1914
  fwd_rxn_rates[INDEX(1914)] = C[INDEX(34)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.0083216641744048e+00 + 2.0 * logT - (2.2982188492987420e+03 / T));
  //rxn 1915
  fwd_rxn_rates[INDEX(1915)] = C[INDEX(30)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(4.7874917427820458e+00 + 2.0 * logT - (3.5039429910925933e+03 / T));
  //rxn 1916
  fwd_rxn_rates[INDEX(1916)] = C[INDEX(60)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.5039429910925933e+03 / T));
  //rxn 1917
  fwd_rxn_rates[INDEX(1917)] = C[INDEX(59)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(4.3944491546724391e+00 + 2.0 * logT - (5.1142455986419136e+03 / T));
  //rxn 1918
  fwd_rxn_rates[INDEX(1918)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(102)] * exp(5.0875963352323836e+00 + 2.0 * logT - (2.6001505882142396e+03 / T));
  //rxn 1919
  fwd_rxn_rates[INDEX(1919)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(99)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.3212491280704739e+03 / T));
  //rxn 1920
  fwd_rxn_rates[INDEX(1920)] = shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.3181199938442161e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1921
  fwd_rxn_rates[INDEX(1921)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(104)] * exp(5.5028895322768108e+00 + 2.0 * logT - (2.9382185240822737e+03 / T));
  //rxn 1922
  fwd_rxn_rates[INDEX(1922)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(108)] * exp(1.2310432660854417e+01 + 1.0 * logT - (1.3421016760663329e+03 / T));
  //rxn 1923
  fwd_rxn_rates[INDEX(1923)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(110)] * exp(5.3181199938442161e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1924
  fwd_rxn_rates[INDEX(1924)] = C[INDEX(23)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.1346212738772472e+00 + 2.0 * logT - (5.2652114680996629e+03 / T));
  //rxn 1925
  fwd_rxn_rates[INDEX(1925)] = C[INDEX(25)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.6280413761795334e+00 + 2.0 * logT - (6.1092314510640454e+03 / T));
  //rxn 1926
  fwd_rxn_rates[INDEX(1926)] = C[INDEX(45)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.6970342476664841e+00 + 2.0 * logT - (6.9647047113246217e+03 / T));
  //rxn 1927
  fwd_rxn_rates[INDEX(1927)] = C[INDEX(46)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(4.9272536851572051e+00 + 2.0 * logT - (5.7066557991767149e+03 / T));
  //rxn 1928
  fwd_rxn_rates[INDEX(1928)] = C[INDEX(79)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.0875963352323836e+00 + 2.0 * logT - (7.3807364543762860e+03 / T));
  //rxn 1929
  fwd_rxn_rates[INDEX(1929)] = C[INDEX(48)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.5482975718158061e+00 + 2.0 * logT - (2.9524042836156536e+03 / T));
  //rxn 1930
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(109)];
  shared_temp[threadIdx.x] = C[INDEX(110)];
  fwd_rxn_rates[INDEX(1930)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.2983173665480363e+00 + 2.0 * logT - (3.5126184963907649e+03 / T));
  //rxn 1931
  fwd_rxn_rates[INDEX(1931)] = C[INDEX(60)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.9052747784384296e+00 + 2.0 * logT - (3.5126184963907649e+03 / T));
  //rxn 1932
  fwd_rxn_rates[INDEX(1932)] = C[INDEX(59)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.9052747784384296e+00 + 2.0 * logT - (5.1229160717444374e+03 / T));
  //rxn 1933
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1933)] = C[INDEX(63)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.5984219589983750e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1934
  fwd_rxn_rates[INDEX(1934)] = C[INDEX(64)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.5984219589983750e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1935
  fwd_rxn_rates[INDEX(1935)] = C[INDEX(62)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.2897155709089976e+00 + 2.0 * logT - (6.4412104301972831e+03 / T));
  //rxn 1936
  fwd_rxn_rates[INDEX(1936)] = C[INDEX(102)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.5984219589983750e+00 + 2.0 * logT - (2.6221362510029367e+03 / T));
  //rxn 1937
  fwd_rxn_rates[INDEX(1937)] = C[INDEX(99)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.9052747784384296e+00 + 2.0 * logT - (3.3212491280704739e+03 / T));
  //rxn 1938
  fwd_rxn_rates[INDEX(1938)] = C[INDEX(98)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.8289456176102075e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1939
  fwd_rxn_rates[INDEX(1939)] = C[INDEX(94)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.5984219589983750e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1940
  fwd_rxn_rates[INDEX(1940)] = C[INDEX(104)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.0137151560428022e+00 + 2.0 * logT - (2.9592782628716295e+03 / T));
  //rxn 1941
  fwd_rxn_rates[INDEX(1941)] = C[INDEX(108)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(1.2821258284620408e+01 + 1.0 * logT - (1.3640873388550297e+03 / T));
  //rxn 1942
  fwd_rxn_rates[INDEX(1942)] = C[INDEX(5)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(1.0064755700132251e+01 + 2.0 * logT - (1.2580489121479068e+03 / T));
  //rxn 1943
  fwd_rxn_rates[INDEX(1943)] = C[INDEX(4)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.7409686230383539e+00 + 2.0 * logT - (2.2141660853803161e+04 / T));
  //rxn 1944
  fwd_rxn_rates[INDEX(1944)] = C[INDEX(9)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(8.9871968206619730e+00 + 2.0 * logT - (7.8072049712031294e+03 / T));
  //rxn 1945
  fwd_rxn_rates[INDEX(1945)] = C[INDEX(22)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.0591231955817966e+00 + 2.0 * logT - (8.9587675912311886e+02 / T));
  //rxn 1946
  fwd_rxn_rates[INDEX(1946)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.6454468976432377e+00 + 2.0 * logT - (5.2738819412021867e+03 / T));
  //rxn 1947
  fwd_rxn_rates[INDEX(1947)] = C[INDEX(25)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(7.1388669999455239e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1948
  fwd_rxn_rates[INDEX(1948)] = C[INDEX(45)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(7.2078598714324755e+00 + 2.0 * logT - (6.9594410346761952e+03 / T));
  //rxn 1949
  fwd_rxn_rates[INDEX(1949)] = C[INDEX(46)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.4380793089231956e+00 + 2.0 * logT - (5.7013921225282875e+03 / T));
  //rxn 1950
  fwd_rxn_rates[INDEX(1950)] = C[INDEX(79)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.5984219589983750e+00 + 2.0 * logT - (7.4027221171649826e+03 / T));
  //rxn 1951
  fwd_rxn_rates[INDEX(1951)] = C[INDEX(48)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.0591231955817966e+00 + 2.0 * logT - (2.9743899464043507e+03 / T));
  //rxn 1952
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(106)];
  shared_temp[threadIdx.x] = C[INDEX(108)];
  fwd_rxn_rates[INDEX(1952)] = C[INDEX(14)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.1737861039019366e+00 + 2.0 * logT - (4.4913000349114591e+03 / T));
  //rxn 1953
  fwd_rxn_rates[INDEX(1953)] = C[INDEX(34)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.9891509171861310e+00 + 2.0 * logT - (3.7543450465665123e+03 / T));
  //rxn 1954
  fwd_rxn_rates[INDEX(1954)] = C[INDEX(30)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.7683209957937720e+00 + 2.0 * logT - (6.2047425244743145e+03 / T));
  //rxn 1955
  fwd_rxn_rates[INDEX(1955)] = C[INDEX(60)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.3752784076841653e+00 + 2.0 * logT - (6.2047425244743145e+03 / T));
  //rxn 1956
  fwd_rxn_rates[INDEX(1956)] = C[INDEX(59)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.3752784076841653e+00 + 2.0 * logT - (8.1167705509175403e+03 / T));
  //rxn 1957
  fwd_rxn_rates[INDEX(1957)] = C[INDEX(63)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.0684255882441107e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1958
  fwd_rxn_rates[INDEX(1958)] = C[INDEX(64)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.0684255882441107e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1959
  fwd_rxn_rates[INDEX(1959)] = C[INDEX(62)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.7597192001547324e+00 + 2.0 * logT - (1.0282235046810787e+04 / T));
  //rxn 1960
  fwd_rxn_rates[INDEX(1960)] = C[INDEX(102)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.0684255882441107e+00 + 2.0 * logT - (4.1302450293206584e+03 / T));
  //rxn 1961
  fwd_rxn_rates[INDEX(1961)] = C[INDEX(99)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.3752784076841653e+00 + 2.0 * logT - (6.6147255683563708e+03 / T));
  //rxn 1962
  fwd_rxn_rates[INDEX(1962)] = C[INDEX(98)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.2989492468559423e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1963
  fwd_rxn_rates[INDEX(1963)] = C[INDEX(94)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.0684255882441107e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1964
  fwd_rxn_rates[INDEX(1964)] = C[INDEX(104)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.4837187852885370e+00 + 2.0 * logT - (4.6149360498017068e+03 / T));
  //rxn 1965
  fwd_rxn_rates[INDEX(1965)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(110)] * exp(6.2989492468559423e+00 + 2.0 * logT - (1.0975581027665039e+04 / T));
  //rxn 1966
  fwd_rxn_rates[INDEX(1966)] = C[INDEX(22)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.5291268248275323e+00 + 2.0 * logT - (2.9287328352846785e+03 / T));
  //rxn 1967
  fwd_rxn_rates[INDEX(1967)] = C[INDEX(23)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.1154505268889734e+00 + 2.0 * logT - (8.2935364874656188e+03 / T));
  //rxn 1968
  fwd_rxn_rates[INDEX(1968)] = C[INDEX(45)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(7.6778635006782103e+00 + 2.0 * logT - (1.1261334257570314e+04 / T));
  //rxn 1969
  fwd_rxn_rates[INDEX(1969)] = C[INDEX(46)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(5.9080829381689313e+00 + 2.0 * logT - (9.8119059127108176e+03 / T));
  //rxn 1970
  fwd_rxn_rates[INDEX(1970)] = C[INDEX(79)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.0684255882441107e+00 + 2.0 * logT - (9.7821404754493997e+03 / T));
  //rxn 1971
  fwd_rxn_rates[INDEX(1971)] = C[INDEX(48)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.5291268248275323e+00 + 2.0 * logT - (4.5645084172071702e+03 / T));
  //rxn 1972
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(3)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(107)];
  shared_temp[threadIdx.x] = C[INDEX(111)];
  fwd_rxn_rates[INDEX(1972)] = shared_temp[threadIdx.x + 3 * blockDim.x] * shared_temp[threadIdx.x] * exp(1.0085809109330082e+01 + 2.0 * logT - (1.4217613331835382e+03 / T));
  //rxn 1973
  fwd_rxn_rates[INDEX(1973)] = C[INDEX(34)] * shared_temp[threadIdx.x] * exp(6.2960037366261856e+00 + 2.0 * logT - (1.3739202491523779e+03 / T));
  //rxn 1974
  fwd_rxn_rates[INDEX(1974)] = C[INDEX(30)] * shared_temp[threadIdx.x] * exp(5.0751738152338266e+00 + 2.0 * logT - (2.7201131002810157e+03 / T));
  //rxn 1975
  fwd_rxn_rates[INDEX(1975)] = C[INDEX(60)] * shared_temp[threadIdx.x] * exp(4.6821312271242199e+00 + 2.0 * logT - (2.7201131002810157e+03 / T));
  //rxn 1976
  fwd_rxn_rates[INDEX(1976)] = C[INDEX(59)] * shared_temp[threadIdx.x] * exp(4.6821312271242199e+00 + 2.0 * logT - (4.2709200586770376e+03 / T));
  //rxn 1977
  fwd_rxn_rates[INDEX(1977)] = C[INDEX(63)] * shared_temp[threadIdx.x] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.6612686850782568e+03 / T));
  //rxn 1978
  fwd_rxn_rates[INDEX(1978)] = C[INDEX(64)] * shared_temp[threadIdx.x] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.6612686850782568e+03 / T));
  //rxn 1979
  fwd_rxn_rates[INDEX(1979)] = C[INDEX(62)] * shared_temp[threadIdx.x] * exp(6.0665720195947870e+00 + 2.0 * logT - (5.6838247275190542e+03 / T));
  //rxn 1980
  fwd_rxn_rates[INDEX(1980)] = C[INDEX(102)] * shared_temp[threadIdx.x] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.6612686850782568e+03 / T));
  //rxn 1981
  fwd_rxn_rates[INDEX(1981)] = C[INDEX(99)] * shared_temp[threadIdx.x] * exp(4.6821312271242199e+00 + 2.0 * logT - (2.6718291830327789e+03 / T));
  //rxn 1982
  fwd_rxn_rates[INDEX(1982)] = C[INDEX(98)] * shared_temp[threadIdx.x] * exp(5.6058020662959978e+00 + 2.0 * logT - (6.1234121784017761e+03 / T));
  //rxn 1983
  fwd_rxn_rates[INDEX(1983)] = C[INDEX(94)] * shared_temp[threadIdx.x] * exp(5.3752784076841653e+00 + 2.0 * logT - (6.1234121784017761e+03 / T));
  //rxn 1984
  fwd_rxn_rates[INDEX(1984)] = C[INDEX(104)] * shared_temp[threadIdx.x] * exp(5.7905716047285916e+00 + 2.0 * logT - (1.9973489036650972e+03 / T));
  //rxn 1985
  fwd_rxn_rates[INDEX(1985)] = C[INDEX(108)] * shared_temp[threadIdx.x] * exp(1.2598114733306197e+01 + 1.0 * logT - (5.3725733622623659e+02 / T));
  //rxn 1986
  fwd_rxn_rates[INDEX(1986)] = C[INDEX(5)] * shared_temp[threadIdx.x] * exp(9.8416121488180401e+00 + 2.0 * logT - (7.0659575199699339e+02 / T));
  //rxn 1987
  fwd_rxn_rates[INDEX(1987)] = C[INDEX(4)] * shared_temp[threadIdx.x] * exp(9.5178250717241433e+00 + 2.0 * logT - (2.1066622833003235e+04 / T));
  //rxn 1988
  fwd_rxn_rates[INDEX(1988)] = C[INDEX(7)] * shared_temp[threadIdx.x] * exp(1.6300417207752275e+01 + 1.0 * logT - (-3.4992378881611603e+02 / T));
  //rxn 1989
  fwd_rxn_rates[INDEX(1989)] = C[INDEX(9)] * shared_temp[threadIdx.x] * exp(8.7640532693477624e+00 + 2.0 * logT - (7.2487469947093759e+03 / T));
  //rxn 1990
  fwd_rxn_rates[INDEX(1990)] = C[INDEX(22)] * shared_temp[threadIdx.x] * exp(5.8359796442675869e+00 + 2.0 * logT - (2.3336807320343672e+02 / T));
  //rxn 1991
  fwd_rxn_rates[INDEX(1991)] = C[INDEX(23)] * shared_temp[threadIdx.x] * exp(5.4223033463290280e+00 + 2.0 * logT - (4.4167983783340605e+03 / T));
  //rxn 1992
  fwd_rxn_rates[INDEX(1992)] = C[INDEX(25)] * shared_temp[threadIdx.x] * exp(6.9157234486313142e+00 + 2.0 * logT - (5.4398185927144950e+03 / T));
  //rxn 1993
  fwd_rxn_rates[INDEX(1993)] = C[INDEX(45)] * shared_temp[threadIdx.x] * exp(6.9847163201182658e+00 + 2.0 * logT - (6.2700453274060883e+03 / T));
  //rxn 1994
  fwd_rxn_rates[INDEX(1994)] = C[INDEX(46)] * shared_temp[threadIdx.x] * exp(5.2149357576089859e+00 + 2.0 * logT - (5.0497328504269699e+03 / T));
  //rxn 1995
  fwd_rxn_rates[INDEX(1995)] = C[INDEX(79)] * shared_temp[threadIdx.x] * exp(5.3752784076841653e+00 + 2.0 * logT - (6.2700453274060883e+03 / T));
  //rxn 1996
  fwd_rxn_rates[INDEX(1996)] = C[INDEX(48)] * shared_temp[threadIdx.x] * exp(5.8359796442675869e+00 + 2.0 * logT - (1.9973489036650972e+03 / T));
  //rxn 1997
  fwd_rxn_rates[INDEX(1997)] = C[INDEX(14)] * shared_temp[threadIdx.x] * exp(5.8861040314501558e+00 + 2.0 * logT - (1.9012389989726457e+03 / T));
  //rxn 1998
  fwd_rxn_rates[INDEX(1998)] = C[INDEX(110)] * shared_temp[threadIdx.x] * exp(6.2146080984221914e+00 + 2.0 * logT - (8.3031228201761860e+03 / T));
  //rxn 1999
  kf = exp(4.0173565998079937e+01 - 0.39 * logT - (5.5519308795694677e+04 / T));
  fwd_rxn_rates[INDEX(1999)] = C[INDEX(126)] * kf;
  if (T <= 860.0) {
    Kc = (-2.9440794580000000e+00 + 1.4995054400000001e+00 * logT + T * (1.4958202300000001e-06 + T * (-9.8793293166666672e-10 + T * (4.0650847083333331e-13 + -7.2769659999999997e-17 * T))) - 2.5473786599999999e+04 / T);
  } else {
    Kc = (-2.9486721579999999e+00 + 1.5003149300000000e+00 * logT + T * (-3.8670341399999998e-07 + T * (1.0655755766666666e-10 + T * (-1.7712649249999999e-14 + 1.2223959550000001e-18 * T))) - 2.5473647400000002e+04 / T);
  }

  if (T <= 1280.0) {
    Kc += (-3.8575027459999998e+00 + 3.1067862499999999e+00 * logT + T * (-6.5482846499999997e-04 + T * (7.6065466999999997e-07 + T * (-2.3354872416666665e-10 + 2.8597877800000003e-14 * T))) - 2.1193422800000000e+04 / T);
  } else {
    Kc += (5.5775860000000002e+00 + 1.5527341200000002e+00 * logT + T * (1.7733779850000000e-03 + T * (-1.8786347500000000e-07 + T * (1.3461208916666668e-11 + -3.4859875700000000e-16 * T))) - 2.1591260100000000e+04 / T);
  }

  if (T <= 700.0) {
    Kc += (3.7113596160000002e+00 + -3.0509114200000003e+00 * logT + T * (1.4033466350000000e-03 + T * (-2.7945090000000001e-06 + T * (1.4631324916666668e-09 + -3.1831739350000002e-13 * T))) - 6.6774198500000002e+03 / T);
  } else {
    Kc += (-4.6712069999999990e+00 + -1.5178180600000002e+00 * logT + T * (-2.9769201050000001e-03 + T * (3.3425295666666670e-07 + T * (-2.6754154666666669e-11 + 9.4403050999999999e-16 * T))) - 6.4627867800000004e+03 / T);
  }

  Kc = 1.2186597134166982e+01 * exp(Kc);
  rev_rxn_rates[INDEX(858)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(132)] * kf / Kc;

}

