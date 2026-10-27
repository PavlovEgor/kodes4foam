# h2o2 — 10 species, 29 reactions

The smallest of the three mechanisms, and the one whose chemistry is entirely about what
the hydrogen cases burn: no carbon and no NOx chemistry carried along unused.

| File | Read by |
|---|---|
| `reactions` | `foamChemistryReader`, via a case's `constant/thermophysicalProperties` |
| `thermo.compressibleGas` | the same, for JANAF and Sutherland coefficients |
| `mechanism.cuh` | symlink to `pyJacSource/h2o2/out/mechanism.cuh`, for [checkKodesMechanism.py](../../../utilities/checkKodesMechanism.py) |

## The mechanism

[`pyJacSource/h2o2/h2o2.inp`](../../../src/kodesChemistryModel/pyJacSource/h2o2/h2o2.inp)
is the H2/O2 submechanism of GRI-Mech 3.0: 9 species, 28 reactions, argon as the only
diluent. It has **no N2**, so on its own it cannot represent air.

The mechanism used here is that same submechanism *plus* N2 — which is exactly
Cantera's bundled `h2o2.yaml` ("Hydrogen-Oxygen submechanism extracted from GRI-Mech
3.0. Modified from the original to include N2"). The two were checked against each
other before use: every reaction of `h2o2.inp` is present, no reaction is missing
except the one N2 third-body variant `H + O2 + N2 <=> HO2 + N2` that only exists once
N2 does, and every shared forward rate constant agrees to the last bit at 1200 K.

N2 is a bath gas here: it appears as a third body and in that one reaction, and is
otherwise chemically inert. pyJac eliminates it from the state vector, so it is
recovered from `sum(Y) = 1` — the same specie `inertSpecie` names.

The CHEMKIN files were written out of Cantera and are kept next to the mechanism as
`h2o2_n2.CKI`, `h2o2_n2.CKT` and `h2o2_n2.TRAN`.

## Regenerating

```bash
source .venv/bin/activate
source /usr/lib/openfoam/openfoam2412/etc/bashrc
P=src/kodesChemistryModel/pyJacSource/h2o2

# pyJac CUDA sources. -ls N2 makes N2 the eliminated specie; without it pyJac
# picks AR, which is a thousand times less abundant and so a worse thing to
# recover a mass fraction from by subtraction.
python -m pyjac -l cuda -i $P/h2o2_n2.CKI -t $P/h2o2_n2.CKT -ls N2 -b $P/out

# OpenFOAM species table, kinetics, thermo and transport
./utilities/chemkinToFoam.py $P/h2o2_n2.CKI $P/h2o2_n2.CKT $P/h2o2_n2.TRAN \
    tutorials/mechanisms/h2o2
```

Note that `yaml2ck` writes `REACTIONS CAL/MOLE MOLE`, and pyJac only accepts `MOLES`
for that unit; the units it defaults to are the same ones, so the keyword is left bare
in the checked-in `.CKI`.
