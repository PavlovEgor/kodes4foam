#include "rates.cuh"
__device__ void eval_rxn_rates_5 (const double T, const double pres, const double * __restrict__ C, double * __restrict__ fwd_rxn_rates, double * __restrict__ rev_rxn_rates) {
  extern volatile __shared__ double shared_temp[];
  register double logT = log(T);

  register double kf;
  register double Kc;

  //rxn 1250
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(61)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(62)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(21)];
  shared_temp[threadIdx.x] = C[INDEX(23)];
  fwd_rxn_rates[INDEX(1250)] = shared_temp[threadIdx.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.4414740933173018e+00 + 2.0 * logT - (5.2385961853142617e+03 / T));
  //rxn 1251
  fwd_rxn_rates[INDEX(1251)] = C[INDEX(25)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.9348941956195880e+00 + 2.0 * logT - (6.1253948634873213e+03 / T));
  //rxn 1252
  fwd_rxn_rates[INDEX(1252)] = C[INDEX(45)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.0038870671065387e+00 + 2.0 * logT - (6.9808681237478977e+03 / T));
  //rxn 1253
  fwd_rxn_rates[INDEX(1253)] = C[INDEX(46)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.2341065045972597e+00 + 2.0 * logT - (5.7228192115999909e+03 / T));
  //rxn 1254
  fwd_rxn_rates[INDEX(1254)] = C[INDEX(48)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.8551503912558607e+00 + 2.0 * logT - (2.8849074433810938e+03 / T));
  //rxn 1255
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(63)];
  fwd_rxn_rates[INDEX(1255)] = C[INDEX(34)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.6028565560662402e+00 + 2.0 * logT - (4.4341242279521612e+03 / T));
  //rxn 1256
  fwd_rxn_rates[INDEX(1256)] = C[INDEX(30)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.3820266346738812e+00 + 2.0 * logT - (6.4728277154573843e+03 / T));
  //rxn 1257
  fwd_rxn_rates[INDEX(1257)] = C[INDEX(60)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.9889840465642745e+00 + 2.0 * logT - (6.4728277154573843e+03 / T));
  //rxn 1258
  fwd_rxn_rates[INDEX(1258)] = C[INDEX(59)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.9889840465642745e+00 + 2.0 * logT - (8.3848507097049624e+03 / T));
  //rxn 1259
  fwd_rxn_rates[INDEX(1259)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(64)] * exp(4.6821312271242199e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1260
  fwd_rxn_rates[INDEX(1260)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(62)] * exp(5.3734248390348425e+00 + 2.0 * logT - (1.0282235046810787e+04 / T));
  //rxn 1261
  fwd_rxn_rates[INDEX(1261)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(102)] * exp(4.6821312271242199e+00 + 2.0 * logT - (4.8100242107063068e+03 / T));
  //rxn 1262
  fwd_rxn_rates[INDEX(1262)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(99)] * exp(3.9889840465642745e+00 + 2.0 * logT - (6.6147255683563708e+03 / T));
  //rxn 1263
  fwd_rxn_rates[INDEX(1263)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(98)] * exp(4.9126548857360524e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1264
  fwd_rxn_rates[INDEX(1264)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(104)] * exp(5.0974244241686471e+00 + 2.0 * logT - (5.2659914584251947e+03 / T));
  //rxn 1265
  fwd_rxn_rates[INDEX(1265)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(108)] * exp(1.1904967552746252e+01 + 1.0 * logT - (2.8721961171727517e+03 / T));
  //rxn 1266
  fwd_rxn_rates[INDEX(1266)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(110)] * exp(4.9126548857360524e+00 + 2.0 * logT - (1.0975581027665039e+04 / T));
  //rxn 1267
  fwd_rxn_rates[INDEX(1267)] = C[INDEX(22)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.1428324637076415e+00 + 2.0 * logT - (3.1968129940721005e+03 / T));
  //rxn 1268
  fwd_rxn_rates[INDEX(1268)] = shared_temp[threadIdx.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.7291561657690826e+00 + 2.0 * logT - (8.5616166462530400e+03 / T));
  //rxn 1269
  fwd_rxn_rates[INDEX(1269)] = C[INDEX(45)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.2915691395583204e+00 + 2.0 * logT - (1.1098572921512266e+04 / T));
  //rxn 1270
  fwd_rxn_rates[INDEX(1270)] = C[INDEX(46)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.5217885770490405e+00 + 2.0 * logT - (9.6491395444571226e+03 / T));
  //rxn 1271
  fwd_rxn_rates[INDEX(1271)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(79)] * exp(4.6821312271242199e+00 + 2.0 * logT - (1.0461919656835049e+04 / T));
  //rxn 1272
  fwd_rxn_rates[INDEX(1272)] = C[INDEX(48)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.1428324637076415e+00 + 2.0 * logT - (5.2442875985928185e+03 / T));
  //rxn 1273
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(64)];
  fwd_rxn_rates[INDEX(1273)] = C[INDEX(34)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.5530344315675633e+00 + 2.0 * logT - (4.4341242279521612e+03 / T));
  //rxn 1274
  fwd_rxn_rates[INDEX(1274)] = C[INDEX(30)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.3322045101752038e+00 + 2.0 * logT - (6.4728277154573843e+03 / T));
  //rxn 1275
  fwd_rxn_rates[INDEX(1275)] = C[INDEX(60)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(2.9391619220655967e+00 + 2.0 * logT - (6.4728277154573843e+03 / T));
  //rxn 1276
  fwd_rxn_rates[INDEX(1276)] = C[INDEX(59)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(2.9391619220655967e+00 + 2.0 * logT - (8.3848507097049624e+03 / T));
  //rxn 1277
  fwd_rxn_rates[INDEX(1277)] = shared_temp[threadIdx.x + 3 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(3.6323091026255421e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1278
  fwd_rxn_rates[INDEX(1278)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(62)] * exp(4.3236027145361646e+00 + 2.0 * logT - (1.0282235046810787e+04 / T));
  //rxn 1279
  fwd_rxn_rates[INDEX(1279)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(102)] * exp(3.6323091026255421e+00 + 2.0 * logT - (4.8100242107063068e+03 / T));
  //rxn 1280
  fwd_rxn_rates[INDEX(1280)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(99)] * exp(2.9391619220655967e+00 + 2.0 * logT - (6.6147255683563708e+03 / T));
  //rxn 1281
  fwd_rxn_rates[INDEX(1281)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(98)] * exp(3.8628327612373745e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1282
  fwd_rxn_rates[INDEX(1282)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(104)] * exp(4.0476022996699692e+00 + 2.0 * logT - (5.2659914584251947e+03 / T));
  //rxn 1283
  fwd_rxn_rates[INDEX(1283)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(108)] * exp(1.0855145428247575e+01 + 1.0 * logT - (2.8721961171727517e+03 / T));
  //rxn 1284
  fwd_rxn_rates[INDEX(1284)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(110)] * exp(3.8628327612373745e+00 + 2.0 * logT - (1.0975581027665039e+04 / T));
  //rxn 1285
  fwd_rxn_rates[INDEX(1285)] = C[INDEX(22)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.0930103392089645e+00 + 2.0 * logT - (3.1968129940721005e+03 / T));
  //rxn 1286
  fwd_rxn_rates[INDEX(1286)] = shared_temp[threadIdx.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.6793340412704048e+00 + 2.0 * logT - (8.5616166462530400e+03 / T));
  //rxn 1287
  fwd_rxn_rates[INDEX(1287)] = C[INDEX(45)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.2417470150596426e+00 + 2.0 * logT - (1.1098572921512266e+04 / T));
  //rxn 1288
  fwd_rxn_rates[INDEX(1288)] = C[INDEX(46)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(3.4719664525503626e+00 + 2.0 * logT - (9.6491395444571226e+03 / T));
  //rxn 1289
  fwd_rxn_rates[INDEX(1289)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(79)] * exp(3.6323091026255421e+00 + 2.0 * logT - (1.0461919656835049e+04 / T));
  //rxn 1290
  fwd_rxn_rates[INDEX(1290)] = C[INDEX(48)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(4.0930103392089645e+00 + 2.0 * logT - (5.2442875985928185e+03 / T));
  //rxn 1291
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(79)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(78)];
  fwd_rxn_rates[INDEX(1291)] = C[INDEX(34)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(8.3109067571684498e+00 + 2.0 * logT - (5.3054338079188556e+03 / T));
  //rxn 1292
  fwd_rxn_rates[INDEX(1292)] = C[INDEX(30)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(7.0900768357760917e+00 + 2.0 * logT - (6.8164411309301668e+03 / T));
  //rxn 1293
  fwd_rxn_rates[INDEX(1293)] = C[INDEX(60)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.6970342476664841e+00 + 2.0 * logT - (6.8164411309301668e+03 / T));
  //rxn 1294
  fwd_rxn_rates[INDEX(1294)] = C[INDEX(59)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(6.6970342476664841e+00 + 2.0 * logT - (8.7284691573733926e+03 / T));
  //rxn 1295
  fwd_rxn_rates[INDEX(1295)] = C[INDEX(63)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(7.3901814282264295e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1296
  fwd_rxn_rates[INDEX(1296)] = C[INDEX(64)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(7.3901814282264295e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1297
  fwd_rxn_rates[INDEX(1297)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(99)] * exp(6.6970342476664841e+00 + 2.0 * logT - (6.6147255683563708e+03 / T));
  //rxn 1298
  fwd_rxn_rates[INDEX(1298)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(98)] * exp(7.6207050868382620e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1299
  fwd_rxn_rates[INDEX(1299)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(94)] * exp(7.3901814282264295e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1300
  fwd_rxn_rates[INDEX(1300)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(104)] * exp(7.8054746252708567e+00 + 2.0 * logT - (6.1004854950267927e+03 / T));
  //rxn 1301
  fwd_rxn_rates[INDEX(1301)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(108)] * exp(1.4613017753848462e+01 + 1.0 * logT - (3.7435056971394461e+03 / T));
  //rxn 1302
  fwd_rxn_rates[INDEX(1302)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(110)] * exp(7.6207050868382620e+00 + 2.0 * logT - (1.0975581027665039e+04 / T));
  //rxn 1303
  fwd_rxn_rates[INDEX(1303)] = C[INDEX(5)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(1.1856515169360305e+01 + 2.0 * logT - (4.0547067404396494e+03 / T));
  //rxn 1304
  fwd_rxn_rates[INDEX(1304)] = C[INDEX(22)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(7.8508826648098520e+00 + 2.0 * logT - (3.5404314417405312e+03 / T));
  //rxn 1305
  fwd_rxn_rates[INDEX(1305)] = shared_temp[threadIdx.x] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(7.4372063668712922e+00 + 2.0 * logT - (8.9052350939214703e+03 / T));
  //rxn 1306
  fwd_rxn_rates[INDEX(1306)] = C[INDEX(25)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(8.9306264691735784e+00 + 2.0 * logT - (9.9064457723609103e+03 / T));
  //rxn 1307
  fwd_rxn_rates[INDEX(1307)] = C[INDEX(45)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(8.9996193406605300e+00 + 2.0 * logT - (1.0889948154312953e+04 / T));
  //rxn 1308
  fwd_rxn_rates[INDEX(1308)] = C[INDEX(46)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(7.2298387781512501e+00 + 2.0 * logT - (9.4405198094534608e+03 / T));
  //rxn 1309
  fwd_rxn_rates[INDEX(1309)] = C[INDEX(48)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(7.8508826648098520e+00 + 2.0 * logT - (6.1155971785595138e+03 / T));
  //rxn 1310
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(77)];
  fwd_rxn_rates[INDEX(1310)] = C[INDEX(30)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.0900768357760917e+00 + 2.0 * logT - (6.8164411309301668e+03 / T));
  //rxn 1311
  fwd_rxn_rates[INDEX(1311)] = C[INDEX(60)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.6970342476664841e+00 + 2.0 * logT - (6.8164411309301668e+03 / T));
  //rxn 1312
  fwd_rxn_rates[INDEX(1312)] = C[INDEX(59)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.6970342476664841e+00 + 2.0 * logT - (8.7284691573733926e+03 / T));
  //rxn 1313
  fwd_rxn_rates[INDEX(1313)] = C[INDEX(63)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.3901814282264295e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1314
  fwd_rxn_rates[INDEX(1314)] = C[INDEX(64)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.3901814282264295e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1315
  fwd_rxn_rates[INDEX(1315)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(102)] * exp(7.3901814282264295e+00 + 2.0 * logT - (5.6813337906730012e+03 / T));
  //rxn 1316
  fwd_rxn_rates[INDEX(1316)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(99)] * exp(6.6970342476664841e+00 + 2.0 * logT - (6.6147255683563708e+03 / T));
  //rxn 1317
  fwd_rxn_rates[INDEX(1317)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(98)] * exp(7.6207050868382620e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1318
  fwd_rxn_rates[INDEX(1318)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(94)] * exp(7.3901814282264295e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1319
  fwd_rxn_rates[INDEX(1319)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(104)] * exp(7.8054746252708567e+00 + 2.0 * logT - (6.1004854950267927e+03 / T));
  //rxn 1320
  fwd_rxn_rates[INDEX(1320)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(108)] * exp(1.4613017753848462e+01 + 1.0 * logT - (3.7435056971394461e+03 / T));
  //rxn 1321
  fwd_rxn_rates[INDEX(1321)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(110)] * exp(7.6207050868382620e+00 + 2.0 * logT - (1.0975581027665039e+04 / T));
  //rxn 1322
  fwd_rxn_rates[INDEX(1322)] = C[INDEX(22)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.8508826648098520e+00 + 2.0 * logT - (3.5404314417405312e+03 / T));
  //rxn 1323
  fwd_rxn_rates[INDEX(1323)] = shared_temp[threadIdx.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.4372063668712922e+00 + 2.0 * logT - (8.9052350939214703e+03 / T));
  //rxn 1324
  fwd_rxn_rates[INDEX(1324)] = C[INDEX(25)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.9306264691735784e+00 + 2.0 * logT - (9.9064457723609103e+03 / T));
  //rxn 1325
  fwd_rxn_rates[INDEX(1325)] = C[INDEX(45)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.9996193406605300e+00 + 2.0 * logT - (1.0889948154312953e+04 / T));
  //rxn 1326
  fwd_rxn_rates[INDEX(1326)] = C[INDEX(46)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.2298387781512501e+00 + 2.0 * logT - (9.4405198094534608e+03 / T));
  //rxn 1327
  fwd_rxn_rates[INDEX(1327)] = C[INDEX(48)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(7.8508826648098520e+00 + 2.0 * logT - (6.1155971785595138e+03 / T));
  //rxn 1328
  fwd_rxn_rates[INDEX(1328)] = C[INDEX(6)] * C[INDEX(110)] * exp(5.8289456176102075e+00 + 2.0 * logT - (1.4020341198019294e+04 / T));
  //rxn 1329
  fwd_rxn_rates[INDEX(1329)] = C[INDEX(6)] * C[INDEX(48)] * exp(6.0591231955817966e+00 + 2.0 * logT - (6.5206436385103025e+03 / T));
  //rxn 1330
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(8)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(9)];
  fwd_rxn_rates[INDEX(1330)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(34)] * exp(3.9934186436321402e+00 + 2.0 * logT - (4.2167786656938404e+02 / T));
  //rxn 1331
  fwd_rxn_rates[INDEX(1331)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(104)] * exp(3.4879865117345461e+00 + 2.0 * logT - (9.4825688362930919e+02 / T));
  //rxn 1332
  fwd_rxn_rates[INDEX(1332)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(3.3032169733019514e+00 + 2.0 * logT - (4.3601509519178644e+03 / T));
  //rxn 1333
  fwd_rxn_rates[INDEX(1333)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(46)] * exp(2.9123506646149400e+00 + 2.0 * logT - (3.1135150595009632e+03 / T));
  //rxn 1334
  fwd_rxn_rates[INDEX(1334)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(48)] * exp(3.5333945512735414e+00 + 2.0 * logT - (9.6931159022301654e+02 / T));
  //rxn 1335
  shared_temp[threadIdx.x] = C[INDEX(24)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(25)];
  fwd_rxn_rates[INDEX(1335)] = shared_temp[threadIdx.x] * C[INDEX(30)] * exp(5.0751738152338266e+00 + 2.0 * logT - (2.6710693214898415e+03 / T));
  //rxn 1336
  fwd_rxn_rates[INDEX(1336)] = shared_temp[threadIdx.x] * C[INDEX(60)] * exp(4.6821312271242199e+00 + 2.0 * logT - (2.6710693214898415e+03 / T));
  //rxn 1337
  fwd_rxn_rates[INDEX(1337)] = shared_temp[threadIdx.x] * C[INDEX(59)] * exp(4.6821312271242199e+00 + 2.0 * logT - (4.2091297283079812e+03 / T));
  //rxn 1338
  fwd_rxn_rates[INDEX(1338)] = shared_temp[threadIdx.x] * C[INDEX(62)] * exp(6.0665720195947870e+00 + 2.0 * logT - (5.5215313856563262e+03 / T));
  //rxn 1339
  fwd_rxn_rates[INDEX(1339)] = shared_temp[threadIdx.x] * C[INDEX(102)] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.7616207307024708e+03 / T));
  //rxn 1340
  fwd_rxn_rates[INDEX(1340)] = shared_temp[threadIdx.x] * C[INDEX(99)] * exp(4.6821312271242199e+00 + 2.0 * logT - (2.5326688445666259e+03 / T));
  //rxn 1341
  fwd_rxn_rates[INDEX(1341)] = shared_temp[threadIdx.x] * C[INDEX(98)] * exp(5.6058020662959978e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1342
  fwd_rxn_rates[INDEX(1342)] = shared_temp[threadIdx.x] * C[INDEX(94)] * exp(5.3752784076841653e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1343
  fwd_rxn_rates[INDEX(1343)] = shared_temp[threadIdx.x] * C[INDEX(104)] * exp(5.7905716047285916e+00 + 2.0 * logT - (2.0845367254725957e+03 / T));
  //rxn 1344
  fwd_rxn_rates[INDEX(1344)] = shared_temp[threadIdx.x] * C[INDEX(110)] * exp(5.6058020662959978e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1345
  fwd_rxn_rates[INDEX(1345)] = shared_temp[threadIdx.x] * C[INDEX(45)] * exp(6.9847163201182658e+00 + 2.0 * logT - (6.0489860047591055e+03 / T));
  //rxn 1346
  fwd_rxn_rates[INDEX(1346)] = shared_temp[threadIdx.x] * C[INDEX(46)] * exp(5.2149357576089859e+00 + 2.0 * logT - (4.8367602661872743e+03 / T));
  //rxn 1347
  fwd_rxn_rates[INDEX(1347)] = shared_temp[threadIdx.x] * C[INDEX(79)] * exp(5.3752784076841653e+00 + 2.0 * logT - (6.3335868618608538e+03 / T));
  //rxn 1348
  fwd_rxn_rates[INDEX(1348)] = shared_temp[threadIdx.x] * C[INDEX(48)] * exp(5.8359796442675869e+00 + 2.0 * logT - (2.0942387986830804e+03 / T));
  //rxn 1349
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(21)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(22)];
  fwd_rxn_rates[INDEX(1349)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(63)] * exp(3.2958368660043291e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1350
  fwd_rxn_rates[INDEX(1350)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(64)] * exp(3.2958368660043291e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1351
  fwd_rxn_rates[INDEX(1351)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(102)] * exp(3.2958368660043291e+00 + 2.0 * logT - (2.5426275597551889e+03 / T));
  //rxn 1352
  fwd_rxn_rates[INDEX(1352)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(104)] * exp(3.7111300630487558e+00 + 2.0 * logT - (2.9664491416708725e+03 / T));
  //rxn 1353
  fwd_rxn_rates[INDEX(1353)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(108)] * exp(1.0518673191626361e+01 + 1.0 * logT - (1.0164984888198602e+03 / T));
  //rxn 1354
  fwd_rxn_rates[INDEX(1354)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(110)] * exp(3.5263605246161616e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1355
  fwd_rxn_rates[INDEX(1355)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(45)] * exp(4.9052747784384296e+00 + 2.0 * logT - (8.8174031510709519e+03 / T));
  //rxn 1356
  fwd_rxn_rates[INDEX(1356)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(46)] * exp(3.1354942159291497e+00 + 2.0 * logT - (7.4838813685854684e+03 / T));
  //rxn 1357
  fwd_rxn_rates[INDEX(1357)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(79)] * exp(3.2958368660043291e+00 + 2.0 * logT - (7.6668318735856656e+03 / T));
  //rxn 1358
  fwd_rxn_rates[INDEX(1358)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(48)] * exp(3.7565381025877511e+00 + 2.0 * logT - (2.9272231765901010e+03 / T));
  //rxn 1359
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(23)];
  fwd_rxn_rates[INDEX(1359)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(34)] * exp(5.3151744836144594e+00 + 2.0 * logT - (3.1983226527666779e+03 / T));
  //rxn 1360
  fwd_rxn_rates[INDEX(1360)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(30)] * exp(4.0943445622221004e+00 + 2.0 * logT - (4.6975546379602847e+03 / T));
  //rxn 1361
  fwd_rxn_rates[INDEX(1361)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(60)] * exp(3.7013019741124933e+00 + 2.0 * logT - (4.6975546379602847e+03 / T));
  //rxn 1362
  fwd_rxn_rates[INDEX(1362)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(59)] * exp(3.7013019741124933e+00 + 2.0 * logT - (6.4268485438162024e+03 / T));
  //rxn 1363
  shared_temp[threadIdx.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1363)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(63)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1364
  fwd_rxn_rates[INDEX(1364)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(64)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1365
  fwd_rxn_rates[INDEX(1365)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(62)] * exp(5.0857427665830608e+00 + 2.0 * logT - (7.9559818355537409e+03 / T));
  //rxn 1366
  fwd_rxn_rates[INDEX(1366)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(102)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.5294209976614129e+03 / T));
  //rxn 1367
  fwd_rxn_rates[INDEX(1367)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(99)] * exp(3.7013019741124933e+00 + 2.0 * logT - (4.6200839859502157e+03 / T));
  //rxn 1368
  fwd_rxn_rates[INDEX(1368)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(98)] * exp(4.6249728132842707e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1369
  fwd_rxn_rates[INDEX(1369)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(94)] * exp(4.3944491546724391e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1370
  fwd_rxn_rates[INDEX(1370)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(104)] * exp(4.8097423517168654e+00 + 2.0 * logT - (3.9115458064328664e+03 / T));
  //rxn 1371
  fwd_rxn_rates[INDEX(1371)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(108)] * exp(1.1617285480294472e+01 + 1.0 * logT - (2.0032919267260838e+03 / T));
  //rxn 1372
  fwd_rxn_rates[INDEX(1372)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(110)] * exp(4.6249728132842707e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1373
  fwd_rxn_rates[INDEX(1373)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(45)] * exp(6.0038870671065387e+00 + 2.0 * logT - (8.5811264687826297e+03 / T));
  //rxn 1374
  fwd_rxn_rates[INDEX(1374)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(46)] * exp(4.2341065045972597e+00 + 2.0 * logT - (7.2476046862971452e+03 / T));
  //rxn 1375
  fwd_rxn_rates[INDEX(1375)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(79)] * exp(4.3944491546724391e+00 + 2.0 * logT - (8.6536253114918891e+03 / T));
  //rxn 1376
  fwd_rxn_rates[INDEX(1376)] = shared_temp[threadIdx.x + 2 * blockDim.x] * C[INDEX(48)] * exp(4.8551503912558607e+00 + 2.0 * logT - (3.9140166144963250e+03 / T));
  //rxn 1377
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(44)];
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(45)];
  fwd_rxn_rates[INDEX(1377)] = C[INDEX(14)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.1929568508902104e+00 + 2.0 * logT - (2.1120225781052036e+03 / T));
  //rxn 1378
  fwd_rxn_rates[INDEX(1378)] = C[INDEX(34)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.0083216641744048e+00 + 2.0 * logT - (1.6054314421614843e+03 / T));
  //rxn 1379
  fwd_rxn_rates[INDEX(1379)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.7874917427820458e+00 + 2.0 * logT - (2.7215573404321613e+03 / T));
  //rxn 1380
  fwd_rxn_rates[INDEX(1380)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(60)] * exp(4.3944491546724391e+00 + 2.0 * logT - (2.7215573404321613e+03 / T));
  //rxn 1381
  fwd_rxn_rates[INDEX(1381)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(59)] * exp(4.3944491546724391e+00 + 2.0 * logT - (4.2596177472503005e+03 / T));
  //rxn 1382
  fwd_rxn_rates[INDEX(1382)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(63)] * exp(5.0875963352323836e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1383
  fwd_rxn_rates[INDEX(1383)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(64)] * exp(5.0875963352323836e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1384
  fwd_rxn_rates[INDEX(1384)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(62)] * exp(5.7788899471430062e+00 + 2.0 * logT - (5.5215313856563262e+03 / T));
  //rxn 1385
  fwd_rxn_rates[INDEX(1385)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(102)] * exp(5.0875963352323836e+00 + 2.0 * logT - (1.8896548845895879e+03 / T));
  //rxn 1386
  fwd_rxn_rates[INDEX(1386)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(99)] * exp(4.3944491546724391e+00 + 2.0 * logT - (2.5326688445666259e+03 / T));
  //rxn 1387
  fwd_rxn_rates[INDEX(1387)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(98)] * exp(5.3181199938442161e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1388
  fwd_rxn_rates[INDEX(1388)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(94)] * exp(5.0875963352323836e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1389
  fwd_rxn_rates[INDEX(1389)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(104)] * exp(5.5028895322768108e+00 + 2.0 * logT - (2.2071612690374764e+03 / T));
  //rxn 1390
  fwd_rxn_rates[INDEX(1390)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(108)] * exp(1.2310432660854417e+01 + 1.0 * logT - (7.9436730849972855e+02 / T));
  //rxn 1391
  fwd_rxn_rates[INDEX(1391)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(5.3181199938442161e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1392
  fwd_rxn_rates[INDEX(1392)] = C[INDEX(22)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.5482975718158061e+00 + 2.0 * logT - (2.6267054846518579e+02 / T));
  //rxn 1393
  fwd_rxn_rates[INDEX(1393)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.1346212738772472e+00 + 2.0 * logT - (4.4044040804515789e+03 / T));
  //rxn 1394
  fwd_rxn_rates[INDEX(1394)] = C[INDEX(25)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(6.6280413761795334e+00 + 2.0 * logT - (5.1935127444985292e+03 / T));
  //rxn 1395
  fwd_rxn_rates[INDEX(1395)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(46)] * exp(4.9272536851572051e+00 + 2.0 * logT - (4.8061091624917026e+03 / T));
  //rxn 1396
  fwd_rxn_rates[INDEX(1396)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(79)] * exp(5.0875963352323836e+00 + 2.0 * logT - (6.4616159835523222e+03 / T));
  //rxn 1397
  fwd_rxn_rates[INDEX(1397)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(48)] * exp(5.5482975718158061e+00 + 2.0 * logT - (2.2222729525701970e+03 / T));
  //rxn 1398
  shared_temp[threadIdx.x] = C[INDEX(46)];
  fwd_rxn_rates[INDEX(1398)] = C[INDEX(14)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.4998096703302650e+00 + 2.0 * logT - (3.9050593062418316e+03 / T));
  //rxn 1399
  fwd_rxn_rates[INDEX(1399)] = C[INDEX(34)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.3151744836144594e+00 + 2.0 * logT - (3.3060971869725649e+03 / T));
  //rxn 1400
  fwd_rxn_rates[INDEX(1400)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.0943445622221004e+00 + 2.0 * logT - (4.7400565624082883e+03 / T));
  //rxn 1401
  fwd_rxn_rates[INDEX(1401)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(60)] * exp(3.7013019741124933e+00 + 2.0 * logT - (4.7400565624082883e+03 / T));
  //rxn 1402
  fwd_rxn_rates[INDEX(1402)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(59)] * exp(3.7013019741124933e+00 + 2.0 * logT - (6.4693504682642078e+03 / T));
  //rxn 1403
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1403)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(63)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1404
  fwd_rxn_rates[INDEX(1404)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(64)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.4709317876378318e+03 / T));
  //rxn 1405
  fwd_rxn_rates[INDEX(1405)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(62)] * exp(5.0857427665830608e+00 + 2.0 * logT - (7.9559818355537409e+03 / T));
  //rxn 1406
  fwd_rxn_rates[INDEX(1406)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(102)] * exp(4.3944491546724391e+00 + 2.0 * logT - (3.6372005640629477e+03 / T));
  //rxn 1407
  fwd_rxn_rates[INDEX(1407)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(99)] * exp(3.7013019741124933e+00 + 2.0 * logT - (4.6200839859502157e+03 / T));
  //rxn 1408
  fwd_rxn_rates[INDEX(1408)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(98)] * exp(4.6249728132842707e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1409
  fwd_rxn_rates[INDEX(1409)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(94)] * exp(4.3944491546724391e+00 + 2.0 * logT - (8.4354947267123880e+03 / T));
  //rxn 1410
  fwd_rxn_rates[INDEX(1410)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(104)] * exp(4.8097423517168654e+00 + 2.0 * logT - (4.0147712357724263e+03 / T));
  //rxn 1411
  fwd_rxn_rates[INDEX(1411)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(108)] * exp(1.1617285480294472e+01 + 1.0 * logT - (2.1110714931276193e+03 / T));
  //rxn 1412
  fwd_rxn_rates[INDEX(1412)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(110)] * exp(4.6249728132842707e+00 + 2.0 * logT - (8.5951310692726602e+03 / T));
  //rxn 1413
  fwd_rxn_rates[INDEX(1413)] = C[INDEX(5)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(8.8607828958063148e+00 + 2.0 * logT - (2.3609552324497336e+03 / T));
  //rxn 1414
  fwd_rxn_rates[INDEX(1414)] = C[INDEX(9)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(7.7832240163360371e+00 + 2.0 * logT - (9.5434735002279958e+03 / T));
  //rxn 1415
  fwd_rxn_rates[INDEX(1415)] = C[INDEX(22)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.8551503912558607e+00 + 2.0 * logT - (1.8633214047605079e+03 / T));
  //rxn 1416
  fwd_rxn_rates[INDEX(1416)] = C[INDEX(23)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.4414740933173018e+00 + 2.0 * logT - (6.6304914373234087e+03 / T));
  //rxn 1417
  fwd_rxn_rates[INDEX(1417)] = C[INDEX(25)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.9348941956195880e+00 + 2.0 * logT - (7.6493600902937560e+03 / T));
  //rxn 1418
  fwd_rxn_rates[INDEX(1418)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.0038870671065387e+00 + 2.0 * logT - (8.5553213694966507e+03 / T));
  //rxn 1419
  fwd_rxn_rates[INDEX(1419)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(79)] * exp(4.3944491546724391e+00 + 2.0 * logT - (8.7613998456977752e+03 / T));
  //rxn 1420
  fwd_rxn_rates[INDEX(1420)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(48)] * exp(4.8551503912558607e+00 + 2.0 * logT - (4.0217961808978603e+03 / T));
  //rxn 1421
  shared_temp[threadIdx.x] = C[INDEX(81)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(83)];
  fwd_rxn_rates[INDEX(1421)] = C[INDEX(3)] * shared_temp[threadIdx.x] * exp(9.2591305361456140e+00 + 2.0 * logT - (2.0128782594366510e+03 / T));
  //rxn 1422
  fwd_rxn_rates[INDEX(1422)] = C[INDEX(14)] * shared_temp[threadIdx.x] * exp(4.6539603501575231e+00 + 2.0 * logT - (2.5160978242958136e+03 / T));
  //rxn 1423
  fwd_rxn_rates[INDEX(1423)] = C[INDEX(34)] * shared_temp[threadIdx.x] * exp(5.4693251634417175e+00 + 2.0 * logT - (1.9625563029507348e+03 / T));
  //rxn 1424
  fwd_rxn_rates[INDEX(1424)] = C[INDEX(30)] * shared_temp[threadIdx.x] * exp(4.2484952420493594e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1425
  fwd_rxn_rates[INDEX(1425)] = C[INDEX(60)] * shared_temp[threadIdx.x] * exp(3.8554526539397518e+00 + 2.0 * logT - (3.3715710845563904e+03 / T));
  //rxn 1426
  fwd_rxn_rates[INDEX(1426)] = C[INDEX(59)] * shared_temp[threadIdx.x] * exp(3.8554526539397518e+00 + 2.0 * logT - (4.9818736921057116e+03 / T));
  //rxn 1427
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1427)] = C[INDEX(63)] * shared_temp[threadIdx.x] * exp(4.5485998344996972e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1428
  fwd_rxn_rates[INDEX(1428)] = C[INDEX(64)] * shared_temp[threadIdx.x] * exp(4.5485998344996972e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1429
  fwd_rxn_rates[INDEX(1429)] = C[INDEX(62)] * shared_temp[threadIdx.x] * exp(5.2398934464103197e+00 + 2.0 * logT - (6.4412104301972831e+03 / T));
  //rxn 1430
  fwd_rxn_rates[INDEX(1430)] = shared_temp[threadIdx.x] * C[INDEX(102)] * exp(4.5485998344996972e+00 + 2.0 * logT - (2.2644880418662324e+03 / T));
  //rxn 1431
  fwd_rxn_rates[INDEX(1431)] = shared_temp[threadIdx.x] * C[INDEX(99)] * exp(3.8554526539397518e+00 + 2.0 * logT - (3.3212491280704739e+03 / T));
  //rxn 1432
  fwd_rxn_rates[INDEX(1432)] = shared_temp[threadIdx.x] * C[INDEX(98)] * exp(4.7791234931115296e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1433
  fwd_rxn_rates[INDEX(1433)] = shared_temp[threadIdx.x] * C[INDEX(94)] * exp(4.5485998344996972e+00 + 2.0 * logT - (6.8941080385705300e+03 / T));
  //rxn 1434
  fwd_rxn_rates[INDEX(1434)] = shared_temp[threadIdx.x] * C[INDEX(104)] * exp(4.9638930315441243e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1435
  fwd_rxn_rates[INDEX(1435)] = shared_temp[threadIdx.x] * C[INDEX(108)] * exp(1.1771436160121729e+01 + 1.0 * logT - (1.0064391297183255e+03 / T));
  //rxn 1436
  fwd_rxn_rates[INDEX(1436)] = shared_temp[threadIdx.x] * C[INDEX(110)] * exp(4.7791234931115296e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1437
  fwd_rxn_rates[INDEX(1437)] = C[INDEX(5)] * shared_temp[threadIdx.x] * exp(9.0149335756335720e+00 + 2.0 * logT - (1.2580489121479068e+03 / T));
  //rxn 1438
  fwd_rxn_rates[INDEX(1438)] = C[INDEX(4)] * shared_temp[threadIdx.x] * exp(8.6911464985396751e+00 + 2.0 * logT - (2.2141660853803161e+04 / T));
  //rxn 1439
  fwd_rxn_rates[INDEX(1439)] = C[INDEX(22)] * shared_temp[threadIdx.x] * exp(5.0093010710831196e+00 + 2.0 * logT - (7.5482934728874409e+02 / T));
  //rxn 1440
  fwd_rxn_rates[INDEX(1440)] = C[INDEX(23)] * shared_temp[threadIdx.x] * exp(4.5956247731445599e+00 + 2.0 * logT - (5.1328395615634599e+03 / T));
  //rxn 1441
  fwd_rxn_rates[INDEX(1441)] = C[INDEX(25)] * shared_temp[threadIdx.x] * exp(6.0890448754468460e+00 + 2.0 * logT - (6.1896006477677020e+03 / T));
  //rxn 1442
  fwd_rxn_rates[INDEX(1442)] = C[INDEX(45)] * shared_temp[threadIdx.x] * exp(6.1580377469337977e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1443
  fwd_rxn_rates[INDEX(1443)] = C[INDEX(46)] * shared_temp[threadIdx.x] * exp(4.3882571844245177e+00 + 2.0 * logT - (5.7870249958803715e+03 / T));
  //rxn 1444
  fwd_rxn_rates[INDEX(1444)] = C[INDEX(79)] * shared_temp[threadIdx.x] * exp(4.5485998344996972e+00 + 2.0 * logT - (7.0450739080282783e+03 / T));
  //rxn 1445
  fwd_rxn_rates[INDEX(1445)] = C[INDEX(48)] * shared_temp[threadIdx.x] * exp(5.0093010710831196e+00 + 2.0 * logT - (2.6167417372676464e+03 / T));
  //rxn 1446
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(10)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(30)];
  fwd_rxn_rates[INDEX(1446)] = C[INDEX(34)] * shared_temp[threadIdx.x] * exp(6.1906432209683597e+00 + 2.0 * logT - (1.2477832330247800e+03 / T));
  //rxn 1447
  fwd_rxn_rates[INDEX(1447)] = shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x] * exp(4.9698132995760007e+00 + 2.0 * logT - (2.5805149607934354e+03 / T));
  //rxn 1448
  fwd_rxn_rates[INDEX(1448)] = C[INDEX(60)] * shared_temp[threadIdx.x] * exp(4.5767707114663931e+00 + 2.0 * logT - (2.5805149607934354e+03 / T));
  //rxn 1449
  fwd_rxn_rates[INDEX(1449)] = C[INDEX(59)] * shared_temp[threadIdx.x] * exp(4.5767707114663931e+00 + 2.0 * logT - (4.1185753676115746e+03 / T));
  //rxn 1450
  fwd_rxn_rates[INDEX(1450)] = C[INDEX(63)] * shared_temp[threadIdx.x] * exp(5.2699178920263385e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1451
  fwd_rxn_rates[INDEX(1451)] = C[INDEX(64)] * shared_temp[threadIdx.x] * exp(5.2699178920263385e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1452
  fwd_rxn_rates[INDEX(1452)] = C[INDEX(62)] * shared_temp[threadIdx.x] * exp(5.9612115039369611e+00 + 2.0 * logT - (5.5215313856563262e+03 / T));
  //rxn 1453
  fwd_rxn_rates[INDEX(1453)] = shared_temp[threadIdx.x] * C[INDEX(102)] * exp(5.2699178920263385e+00 + 2.0 * logT - (1.5320066754528837e+03 / T));
  //rxn 1454
  fwd_rxn_rates[INDEX(1454)] = shared_temp[threadIdx.x] * C[INDEX(99)] * exp(4.5767707114663931e+00 + 2.0 * logT - (2.5326688445666259e+03 / T));
  //rxn 1455
  fwd_rxn_rates[INDEX(1455)] = shared_temp[threadIdx.x] * C[INDEX(98)] * exp(5.5004415506381710e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1456
  fwd_rxn_rates[INDEX(1456)] = shared_temp[threadIdx.x] * C[INDEX(94)] * exp(5.2699178920263385e+00 + 2.0 * logT - (5.9582655816062961e+03 / T));
  //rxn 1457
  fwd_rxn_rates[INDEX(1457)] = shared_temp[threadIdx.x] * C[INDEX(104)] * exp(5.6852110890707657e+00 + 2.0 * logT - (1.8646197112378445e+03 / T));
  //rxn 1458
  fwd_rxn_rates[INDEX(1458)] = shared_temp[threadIdx.x] * C[INDEX(108)] * exp(1.2492754217648372e+01 + 1.0 * logT - (4.3671909936302438e+02 / T));
  //rxn 1459
  fwd_rxn_rates[INDEX(1459)] = shared_temp[threadIdx.x] * C[INDEX(110)] * exp(5.5004415506381710e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1460
  fwd_rxn_rates[INDEX(1460)] = C[INDEX(5)] * shared_temp[threadIdx.x] * exp(9.7362516331602151e+00 + 2.0 * logT - (5.8842470158111621e+02 / T));
  //rxn 1461
  fwd_rxn_rates[INDEX(1461)] = C[INDEX(4)] * shared_temp[threadIdx.x] * exp(9.4124645560663165e+00 + 2.0 * logT - (2.0836253948406356e+04 / T));
  //rxn 1462
  fwd_rxn_rates[INDEX(1462)] = C[INDEX(22)] * shared_temp[threadIdx.x] * exp(5.7306191286097610e+00 + 2.0 * logT - (1.2162816882645963e+02 / T));
  //rxn 1463
  fwd_rxn_rates[INDEX(1463)] = C[INDEX(23)] * shared_temp[threadIdx.x] * exp(5.3169428306712012e+00 + 2.0 * logT - (4.2633617008128531e+03 / T));
  //rxn 1464
  fwd_rxn_rates[INDEX(1464)] = C[INDEX(25)] * shared_temp[threadIdx.x] * exp(6.8103629329734874e+00 + 2.0 * logT - (5.2791506500462619e+03 / T));
  //rxn 1465
  fwd_rxn_rates[INDEX(1465)] = C[INDEX(45)] * shared_temp[threadIdx.x] * exp(6.8793558044604390e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1466
  fwd_rxn_rates[INDEX(1466)] = C[INDEX(46)] * shared_temp[threadIdx.x] * exp(5.1095752419511591e+00 + 2.0 * logT - (4.8917420358437867e+03 / T));
  //rxn 1467
  fwd_rxn_rates[INDEX(1467)] = C[INDEX(79)] * shared_temp[threadIdx.x] * exp(5.2699178920263385e+00 + 2.0 * logT - (6.1039677744156179e+03 / T));
  //rxn 1468
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(47)];
  fwd_rxn_rates[INDEX(1468)] = C[INDEX(48)] * shared_temp[threadIdx.x] * exp(5.7306191286097610e+00 + 2.0 * logT - (1.8646197112378445e+03 / T));
  //rxn 1469
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(48)];
  fwd_rxn_rates[INDEX(1469)] = C[INDEX(3)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(1.0085809109330082e+01 + 2.0 * logT - (4.7583838189604594e+03 / T));
  //rxn 1470
  fwd_rxn_rates[INDEX(1470)] = C[INDEX(14)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.4806389233419912e+00 + 2.0 * logT - (5.2208979532181647e+03 / T));
  //rxn 1471
  fwd_rxn_rates[INDEX(1471)] = C[INDEX(34)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.2960037366261856e+00 + 2.0 * logT - (4.5161339204372589e+03 / T));
  //rxn 1472
  fwd_rxn_rates[INDEX(1472)] = shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.0751738152338266e+00 + 2.0 * logT - (6.5051646046952346e+03 / T));
  //rxn 1473
  fwd_rxn_rates[INDEX(1473)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(60)] * exp(4.6821312271242199e+00 + 2.0 * logT - (6.5051646046952346e+03 / T));
  //rxn 1474
  fwd_rxn_rates[INDEX(1474)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(59)] * exp(4.6821312271242199e+00 + 2.0 * logT - (8.4171926311384614e+03 / T));
  //rxn 1475
  shared_temp[threadIdx.x] = C[INDEX(61)];
  fwd_rxn_rates[INDEX(1475)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(63)] * exp(5.3752784076841653e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1476
  fwd_rxn_rates[INDEX(1476)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(64)] * exp(5.3752784076841653e+00 + 2.0 * logT - (5.3236855815362978e+03 / T));
  //rxn 1477
  fwd_rxn_rates[INDEX(1477)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(62)] * exp(6.0665720195947870e+00 + 2.0 * logT - (1.0282235046810787e+04 / T));
  //rxn 1478
  fwd_rxn_rates[INDEX(1478)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(102)] * exp(5.3752784076841653e+00 + 2.0 * logT - (4.8920339031914045e+03 / T));
  //rxn 1479
  fwd_rxn_rates[INDEX(1479)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(99)] * exp(4.6821312271242199e+00 + 2.0 * logT - (6.6147255683563708e+03 / T));
  //rxn 1480
  fwd_rxn_rates[INDEX(1480)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(98)] * exp(5.6058020662959978e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1481
  fwd_rxn_rates[INDEX(1481)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(94)] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.0802629495418592e+04 / T));
  //rxn 1482
  fwd_rxn_rates[INDEX(1482)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(104)] * exp(5.7905716047285916e+00 + 2.0 * logT - (5.3445339681084124e+03 / T));
  //rxn 1483
  fwd_rxn_rates[INDEX(1483)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(108)] * exp(1.2598114733306197e+01 + 1.0 * logT - (2.9542058096578494e+03 / T));
  //rxn 1484
  fwd_rxn_rates[INDEX(1484)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(110)] * exp(5.6058020662959978e+00 + 2.0 * logT - (1.0975581027665039e+04 / T));
  //rxn 1485
  fwd_rxn_rates[INDEX(1485)] = C[INDEX(5)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(9.8416121488180401e+00 + 2.0 * logT - (4.0547067404396494e+03 / T));
  //rxn 1486
  fwd_rxn_rates[INDEX(1486)] = C[INDEX(4)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(9.5178250717241433e+00 + 2.0 * logT - (2.7593647295595951e+04 / T));
  //rxn 1487
  fwd_rxn_rates[INDEX(1487)] = C[INDEX(7)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(1.6300417207752275e+01 + 1.0 * logT - (1.1613552727602187e+03 / T));
  //rxn 1488
  fwd_rxn_rates[INDEX(1488)] = C[INDEX(9)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(8.7640532693477624e+00 + 2.0 * logT - (1.2417556690768970e+04 / T));
  //rxn 1489
  fwd_rxn_rates[INDEX(1489)] = C[INDEX(22)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.8359796442675869e+00 + 2.0 * logT - (3.2291549155055986e+03 / T));
  //rxn 1490
  fwd_rxn_rates[INDEX(1490)] = C[INDEX(23)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.4223033463290280e+00 + 2.0 * logT - (8.5939585676865390e+03 / T));
  //rxn 1491
  fwd_rxn_rates[INDEX(1491)] = C[INDEX(25)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.9157234486313142e+00 + 2.0 * logT - (1.0095429879943767e+04 / T));
  //rxn 1492
  fwd_rxn_rates[INDEX(1492)] = C[INDEX(45)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(6.9847163201182658e+00 + 2.0 * logT - (1.1078937294091462e+04 / T));
  //rxn 1493
  fwd_rxn_rates[INDEX(1493)] = C[INDEX(46)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(5.2149357576089859e+00 + 2.0 * logT - (9.6295039170363179e+03 / T));
  //rxn 1494
  fwd_rxn_rates[INDEX(1494)] = shared_temp[threadIdx.x + 3 * blockDim.x] * C[INDEX(79)] * exp(5.3752784076841653e+00 + 2.0 * logT - (1.0543929349320146e+04 / T));
  //rxn 1495
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(52)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(10)];
  shared_temp[threadIdx.x] = C[INDEX(23)];
  fwd_rxn_rates[INDEX(1495)] = C[INDEX(3)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(9.3926619287701367e+00 + 2.0 * logT - (1.2950909043171898e+03 / T));
  //rxn 1496
  fwd_rxn_rates[INDEX(1496)] = C[INDEX(14)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.7874917427820458e+00 + 2.0 * logT - (1.7694810203055713e+03 / T));
  //rxn 1497
  fwd_rxn_rates[INDEX(1497)] = C[INDEX(34)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(5.6028565560662402e+00 + 2.0 * logT - (1.2477832330247800e+03 / T));
  //rxn 1498
  fwd_rxn_rates[INDEX(1498)] = C[INDEX(30)] * shared_temp[threadIdx.x + 1 * blockDim.x] * exp(4.3820266346738812e+00 + 2.0 * logT - (2.5805149607934354e+03 / T));
  //rxn 1499
  fwd_rxn_rates[INDEX(1499)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(60)] * exp(3.9889840465642745e+00 + 2.0 * logT - (2.5805149607934354e+03 / T));
}

