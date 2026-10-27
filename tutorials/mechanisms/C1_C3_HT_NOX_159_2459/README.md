# C1_C3_HT_NOX_159_2459 — 159 species, 2459 reactions

The largest of the three mechanisms, and the one the GPU is actually for: at 159 species
the KODES state vector is sixteen times the h2o2 one and the Jacobian two hundred and
fifty times the size, which is the regime where the analytical-Jacobian,
batched-integration approach is supposed to pay.

It is CRECK's C1-C3 high-temperature mechanism with soot and NOx chemistry, not a
hydrogen mechanism, but it contains the H2/O2 and NOx subsets, so a hydrogen-air flame is
inside its range of validity and the carbon species stay at zero.

| File | Read by |
|---|---|
| `reactions` | `foamChemistryReader`, via a case's `constant/thermophysicalProperties` |
| `thermo.compressibleGas` | the same, for JANAF and Sutherland coefficients |
| `mechanism.cuh` | symlink to `pyJacSource/C1_C3_HT_NOX_159_2459/out/mechanism.cuh`, for [checkKodesMechanism.py](../../../utilities/checkKodesMechanism.py) |

## The mechanism

[`external/Kinetic-Mechanisms/Gas-Phase/CoreMechanism_C0-C4/Soot-NOx/C1_C3_HT_NOX_159_2459`](../../../external/Kinetic-Mechanisms/Gas-Phase/CoreMechanism_C0-C4/Soot-NOx/C1_C3_HT_NOX_159_2459),
via the `Kinetic-Mechanisms` submodule. The pyJac CUDA output is already in the
repository at `pyJacSource/C1_C3_HT_NOX_159_2459/out` — 282 generated sources, against
23 for GRI-Mech.

N2 is the last specie in pyJac's ordering and so the one eliminated from the state
vector — the same specie `inertSpecie` names.

## PLOG: the two paths do not solve the same kinetics

236 of the 2459 reactions are PLOG (pressure-dependent Arrhenius), with 1291 pressure
brackets between them. OpenFOAM's `chemkinReader` cannot parse PLOG at all — it reads
the keyword as an unknown third-body specie — so
[utilities/chemkinToFoam.py](../../../utilities/chemkinToFoam.py) strips those lines,
and each affected reaction falls back to its own leading, high-pressure-limit Arrhenius
parameters.

pyJac handles PLOG properly, so:

- the **KODES path** integrates the mechanism with its pressure dependence intact;
- the **`standard` reference path** integrates it without, for those 236 reactions.

The reaction *counts* still agree — 2459 either way — so
[checkKodesMechanism.py](../../../utilities/checkKodesMechanism.py) and the specie
mapping are unaffected. But the two paths are not a like-for-like comparison of results
on this mechanism, only of cost. For a numerical comparison, use `grimech` or `h2o2`,
neither of which has a single PLOG reaction.

(Generating this mechanism's Jacobian also needed two fixes to the PLOG code path in
the pyJac fork, described in
[`pyJacSource/C1_C3_HT_NOX_159_2459/README.md`](../../../src/kodesChemistryModel/pyJacSource/C1_C3_HT_NOX_159_2459/README.md).)

## The CPU reference path is fragile and slow

On a Cantera-profile premixed flame started from cold 298 K gas, OpenFOAM's `seulex`
died inside the very first chemistry solve: a floating-point trap in
`ReversibleReaction<..., thirdBodyArrheniusReactionRate>::kr`, and with
`FOAM_SIGFPE=false` a "Singular matrix" abort in the LU decomposition instead. That is
a property of OpenFOAM's own integrator on a 2459-reaction mechanism, not of the
conversion: the reverse rate constants stay finite in double precision over the whole
200-3500 K range when evaluated with Cantera on exactly the same PLOG-stripped input.

The cases here keep `seulex`, so that `standard` and `kodes` runs read the same
`odeCoeffs`. If a `standard` run on this mechanism stops the same way, set
`odeCoeffs/solver Rosenbrock34;` — it only affects `standard`, since
`kodesChemistryModel` takes its integrator from `kodesSetting/integrationMethod`.

## Regenerating

```bash
source .venv/bin/activate
source /usr/lib/openfoam/openfoam2412/etc/bashrc
P=src/kodesChemistryModel/pyJacSource/C1_C3_HT_NOX_159_2459
M=external/Kinetic-Mechanisms/Gas-Phase/CoreMechanism_C0-C4/Soot-NOx/C1_C3_HT_NOX_159_2459

# The pyJac CUDA sources in $P/out are already in the repository. To rebuild:
python -m pyjac -l cuda -i $M/kinetics.CHEMKIN.CKI -t $M/thermo.CHEMKIN.CKT -b $P/out

./utilities/chemkinToFoam.py $M/kinetics.CHEMKIN.CKI $M/thermo.CHEMKIN.CKT $M/TOT2003.TRAN \
    tutorials/mechanisms/C1_C3_HT_NOX_159_2459
```
