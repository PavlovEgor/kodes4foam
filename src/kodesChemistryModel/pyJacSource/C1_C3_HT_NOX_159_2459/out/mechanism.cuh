#ifndef MECHANISM_cuh
#define MECHANISM_cuh

#ifdef __GNUG__
#include <cuda.h>
#include <cuda_runtime.h>
#include "launch_bounds.cuh"
#include "gpu_macros.cuh"
#endif

struct mechanism_memory {
  double * y;
  double * dy;
  double * conc;
  double * fwd_rates;
  double * rev_rates;
  double * spec_rates;
  double * cp;
  double * h;
  double * dBdT;
  double * jac;
  double * var;
  double * J_nplusjplus;
  double * pres_mod;
};

//last_spec 1
/* Species Indexes
0  AR
1  HE
2  H2
3  H
4  O2
5  O
6  H2O
7  OH
8  H2O2
9  HO2
10  CO
11  CO2
12  HOCO
13  CH4
14  CH3
15  CH2
16  CH2(S)
17  C
18  CH
19  CH3O2H
20  CH3O2
21  CH3OH
22  CH3O
23  CH2OH
24  CH2O
25  HCO
26  HO2CHO
27  HOCHO
28  OCHO
29  C2H6
30  C2H5
31  C2H5O2H
32  C2H5O2
33  C2H4
34  C2H3
35  C2H2
36  C2H
37  C2H5OH
38  C2H5O
39  PC2H4OH
40  SC2H4OH
41  C2H4O2H
42  C2H4O1-2
43  C2H3O1-2
44  CH3CHO
45  CH3CO
46  CH2CHO
47  CH2CO
48  HCCO
49  HCOOH
50  CH3CO3
51  CH3CO3H
52  CH2OHCHO
53  CHOCHO
54  O2C2H4O2H
55  HO2CH2CHO
56  CH3OCHO
57  CH3OCO
58  C3H8
59  IC3H7
60  NC3H7
61  C3H6
62  C3H5-A
63  C3H5-S
64  C3H5-T
65  C3H5O
66  C3H6O
67  CH3CHCHO
68  AC4H7OOH
69  CH3CHCO
70  AC3H5OOH
71  C3H6OH1-2
72  C3H6OH2-1
73  HOC3H6O2
74  SC3H5OH
75  C3H5OH
76  CH2CCH2OH
77  C3H4-P
78  C3H4-A
79  C3H3
80  C3H2
81  C2H5CHO
82  CH2CH2CHO
83  RALD3
84  C2H3CHO
85  CH3COCH3
86  CH3COCH2
87  NC4H10
88  PC4H9
89  SC4H9
90  IC4H10
91  IC4H9
92  TC4H9
93  IC4H8
94  IC4H7
95  IC4H7O
96  C4H8-1
97  C4H8-2
98  C4H71-3
99  C4H71-4
100  C4H71-O
101  C4H6
102  C4H5
103  C4H4
104  C4H3
105  C4H2
106  C6H6
107  FULVENE
108  C6H5
109  C5H6
110  C5H5
111  MCPTD
112  C10H8
113  NO
114  N2O
115  NO2
116  HNO
117  HNO2
118  HONO
119  HONO2
120  N2H2
121  H2NN
122  HNNO
123  NH2NO
124  NH2OH
125  HNOH
126  NH3
127  N2H4
128  N
129  NO3
130  NH
131  NNH
132  NH2
133  H2NO
134  N2H3
135  HCN
136  HNC
137  HNCO
138  HCNO
139  HOCN
140  CH2NO
141  CH3NO
142  CH3NO2
143  CH3ONO
144  CH3ONO2
145  CH3CN
146  CN
147  NCN
148  NCO
149  HNCN
150  H2CN
151  HCNH
152  C2N2
153  CH2CN
154  CH2NH
155  CH3NH2
156  CH2NH2
157  CH3NH
158  N2
*/

//Number of species
#define NSP 159
//Number of variables. NN = NSP + 1 (temperature)
#define NN 160
//Number of forward reactions
#define FWD_RATES 2459
//Number of reversible reactions
#define REV_RATES 1314
//Number of reactions with pressure modified rates
#define PRES_MOD_RATES 67

//Must be implemented by user on a per mechanism basis in mechanism.cu
void set_same_initial_conditions(int, double**, double**);

#if defined (RATES_TEST) || defined (PROFILER)
    void write_jacobian_and_rates_output(int NUM);
#endif
//apply masking of ICs for cache optimized mechanisms
void apply_mask(double*);
void apply_reverse_mask(double*);
#endif

