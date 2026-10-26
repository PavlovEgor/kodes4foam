# h2o2 — 10 species, 29 reactions

The smallest of the three sets, and the one whose chemistry actually describes what the
case burns: a hydrogen-air flame on a hydrogen-oxygen mechanism, with no carbon and no
NOx chemistry carried along unused.

This set holds all four upstream cases:

| Case | What it is | Cells |
|---|---|---|
| [2D_planar_flame](2D_planar_flame/) | laminar freely propagating premixed flame | 10 000 |
| [2D_jet_flame](2D_jet_flame/) | axisymmetric laminar non-premixed jet flame, conjugate nozzle | 249 100 |
| [3D_HYLON_flame_A](3D_HYLON_flame_A/) | turbulent non-premixed coaxial swirl flame, LES | 156 000 |
| [3D_AHEAD](3D_AHEAD/) | turbulent premixed swirl flame, LES | 207 616 |

They are the same four in every set; only the mechanism differs. What each case is,
what was changed to port it, and which of the meshes are reconstructions rather than
the published geometry, is in [../MECHANISM_SETS.md](../MECHANISM_SETS.md) — read the
reconstruction section before comparing a 3D result to a published number.

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
source venv/bin/activate
source /usr/lib/openfoam/openfoam2412/etc/bashrc
P=src/kodesChemistryModel/pyJacSource/h2o2

# pyJac CUDA sources. -ls N2 makes N2 the eliminated specie; without it pyJac
# picks AR, which is a thousand times less abundant and so a worse thing to
# recover a mass fraction from by subtraction.
python -m pyjac -l cuda -i $P/h2o2_n2.CKI -t $P/h2o2_n2.CKT -ls N2 -b $P/out

# Build configuration for that mechanism
# OpenFOAM species table, kinetics, thermo and transport
./utilities/chemkinToFoam.py $P/h2o2_n2.CKI $P/h2o2_n2.CKT $P/h2o2_n2.TRAN \
    tutorials/DLBFoam-Hydrogen-Tutorials/h2o2/mech/thermo
cp $P/out/mechanism.cuh tutorials/DLBFoam-Hydrogen-Tutorials/h2o2/mech/
```

Note that `yaml2ck` writes `REACTIONS CAL/MOLE MOLE`, and pyJac only accepts `MOLES`
for that unit; the units it defaults to are the same ones, so the keyword is left bare
in the checked-in `.CKI`.

## The initial condition of 2D_planar_flame

The `*.dat` profiles in `2D_planar_flame/0.orig` are a Cantera 1D solution of a lean
(phi = 0.5) hydrogen-air flame on the *Capurso* mechanism, and so carry six species
this mechanism does not have — N, NO, N2O, NH, NH2 and NNH. Their field files were
removed, which leaves those mass fractions at `Ydefault`, i.e. zero.

That loses nothing worth keeping: the largest of the six peaks at 1.3e-6, and the
resulting mass-fraction sum is short of one by 7.8e-07, which
`multiComponentMixture::correctMassFractions()` normalises away at start-up with a
warning. The nine remaining species — H2, H, O, O2, OH, H2O, HO2, H2O2, N2 — carry the
profile; AR starts at zero and stays there.


The other three cases set their species uniformly from the streams that feed them --
premixed phi = 0.6 hydrogen-air for `3D_AHEAD`, pure H2 against air for `2D_jet_flame`
and `3D_HYLON_flame_A` -- so only H2, O2 and N2 have fields of their own there and every
other specie of the mechanism starts at `Ydefault`, i.e. zero. That is the same in all
three sets, which is why those three cases are byte-identical between sets apart from
the `mechanism` named in `kodesSetting`.

## Status

All four cases run. Verified with the stock `standard` chemistry model on CPU -- this
machine has no GPU and no nvcc, so `libkodes_h2o2` was never built and the
KODES path itself is unverified. What was checked offline instead is the part that fails
first and loudest when a mechanism and a case disagree: the specie mapping, via
[checkKodesMechanism.py](../../../utilities/checkKodesMechanism.py).

| Case | Checked |
|---|---|
| `2D_planar_flame` | `./Allrun` end to end; 10 species and 29 reactions selected; the flame holds at `min/max(T) = 298, 1648 K`, the adiabatic temperature of the phi = 0.5 initial condition |
| `2D_jet_flame` | `./Allrun` end to end on the coarse `kN 1` mesh; both regions solve, the conjugate coupling holds the nozzle near ambient, and the jet ignites -- 300 to 3085 K over the 0.1 ms ignition pulse |
| `3D_AHEAD` | mesh, `topoSet` and the first steps on the full 207 616-cell mesh; ignition to 2263 K confirmed on a `--refine 0.4` mesh, which reaches it inside a CPU budget |
| `3D_HYLON_flame_A` | the same: full mesh starts and runs; ignition to 1966 K confirmed on a `--refine 0.4` mesh |

The 3D cases were only *ignited* on the coarse meshes because at 200k cells a
`standard`-chemistry CPU run needs thousands of steps to reach the 2-4 ms ignition time.
That is the situation these sets exist to change.
