# C1_C3_HT_NOX_159_2459 — 159 species, 2459 reactions

The largest of the three sets, and the one the GPU is actually for: at 159 species the
KODES state vector is sixteen times the h2o2 one and the Jacobian two hundred and fifty
times the size, which is the regime where the analytical-Jacobian, batched-integration
approach is supposed to pay.

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

Like GRI-Mech, it is not a hydrogen mechanism — it is CRECK's C1-C3 high-temperature
mechanism with soot and NOx chemistry — but it contains the H2/O2 and NOx subsets, so
the hydrogen-air flame is inside its range of validity and the carbon species stay at
zero.

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
on this mechanism, only of cost. For a numerical comparison, use the `grimech` or
`h2o2` set, neither of which has a single PLOG reaction.

(Generating this mechanism's Jacobian also needed two fixes to the PLOG code path in
the pyJac fork, described in
[`pyJacSource/C1_C3_HT_NOX_159_2459/README.md`](../../../src/kodesChemistryModel/pyJacSource/C1_C3_HT_NOX_159_2459/README.md).)

## Regenerating

```bash
source venv/bin/activate
source /usr/lib/openfoam/openfoam2412/etc/bashrc
P=src/kodesChemistryModel/pyJacSource/C1_C3_HT_NOX_159_2459
M=external/Kinetic-Mechanisms/Gas-Phase/CoreMechanism_C0-C4/Soot-NOx/C1_C3_HT_NOX_159_2459

# The pyJac CUDA sources in $P/out are already in the repository. To rebuild:
python -m pyjac -l cuda -i $M/kinetics.CHEMKIN.CKI -t $M/thermo.CHEMKIN.CKT -b $P/out

./utilities/chemkinToFoam.py $M/kinetics.CHEMKIN.CKI $M/thermo.CHEMKIN.CKT $M/TOT2003.TRAN \
    tutorials/DLBFoam-Hydrogen-Tutorials/C1_C3_HT_NOX_159_2459/mech/thermo
cp $P/out/mechanism.cuh tutorials/DLBFoam-Hydrogen-Tutorials/C1_C3_HT_NOX_159_2459/mech/
```

The `mech/thermo/reactions` and `mech/thermo/thermo.compressibleGas` this produces are
byte-identical to the ones already committed under
[`../mech/thermo`](../mech/thermo), which is where the upstream tutorial's own
mechanism files were replaced by this mechanism in an earlier commit.

## The initial condition of 2D_planar_flame

All 15 species the Cantera profiles in `2D_planar_flame/0.orig` cover — H2, H, O, O2,
OH, H2O, HO2, H2O2, N, NH, NH2, NNH, N2O, NO, N2 — are in this mechanism, so the
initial condition transfers whole. The remaining 144 species start at `Ydefault`,
i.e. zero.

The profiles are a Cantera 1D solution of a lean (phi = 0.5) hydrogen-air flame on the
*Capurso* mechanism, so they are an initial condition rather than a converged solution
of this mechanism.


The other three cases set their species uniformly from the streams that feed them --
premixed phi = 0.6 hydrogen-air for `3D_AHEAD`, pure H2 against air for `2D_jet_flame`
and `3D_HYLON_flame_A` -- so only H2, O2 and N2 have fields of their own there and every
other specie of the mechanism starts at `Ydefault`, i.e. zero. That is the same in all
three sets, which is why those three cases are byte-identical between sets apart from
the `mechanism` named in `kodesSetting`.

## The CPU reference path is fragile and slow

`constant/chemistryProperties` selects `Rosenbrock34` rather than `seulex` in
`odeCoeffs`, and that entry only affects the `standard` reference model —
`kodesChemistryModel` takes its integrator from `kodesSetting`'s `integrationMethod`.

With `seulex`, `standard` dies inside the very first chemistry solve: a floating-point
trap in `ReversibleReaction<..., thirdBodyArrheniusReactionRate>::kr`, and with
`FOAM_SIGFPE=false` a "Singular matrix" abort in the LU decomposition instead. This is
a property of OpenFOAM's own integrator on a 2459-reaction mechanism started from cold
298 K gas, not of the conversion: the reverse rate constants stay finite in double
precision over the whole 200-3500 K range when evaluated with Cantera on exactly the
same PLOG-stripped input.

`Rosenbrock34` gets through it, but slowly — this is a 159-species dense Jacobian on
10 000 cells, and a single time step costs the better part of an hour on one core. Take
the reference for what it is worth on a small cut-down mesh; it is not a practical way
to run this case.

## Status

All four cases run. `blockMesh` and the initial conditions are fine everywhere, and
`standard` selects 159 species and 2459 reactions.

`2D_planar_flame` was run end to end with `Rosenbrock34`: the first chemistry solve
completes with the flame at `min/max(T) = 298, 1648 K` -- the same temperature the
`h2o2` and `grimech` sets reach, which is what the shared initial condition should give.
The other three cases are byte-identical to the `h2o2` set's apart from the library
named in `controlDict`, the mechanism the `mech` symlink reaches, and the `odeCoeffs`
solver above, and those were run there -- see [../h2o2/README.md](../h2o2/README.md).

The KODES path cannot be exercised on this machine: it has no GPU and no nvcc, so
`libkodes_C1_C3_HT_NOX_159_2459` was never built. What was checked
offline is the part that fails first and loudest when a mechanism and a case disagree --
the specie mapping, via
[checkKodesMechanism.py](../../../utilities/checkKodesMechanism.py).
