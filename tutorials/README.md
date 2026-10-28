# Tutorials

Six `reactingFoam` cases, each runnable with every mechanism it has initial conditions
for and with either chemistry model. The point is to be able to time
[`kodesChemistryModel`](../src/kodesChemistryModel), the GPU chemistry model, against the
chemistry model OpenFOAM ships, on the same flame, the same mesh and the same tolerances, so that a difference in wall clock is a difference in
the integrator and nothing else.

## Layout

```
tutorials/
  mechanisms/<mechanism>/       reactions, thermo.compressibleGas, mechanism.cuh -> pyJacSource
  geometry/                     the two 3D burner geometries, as JSON
  <case>/
    0.<mechanism>/              initial conditions, one directory per mechanism
    constant/<mechanism>        symlink to ../../mechanisms/<mechanism>, one per 0.<mechanism>
    constant/, system/          everything else, shared by every mechanism and variant
    Allrun, Allclean
  Allrun, Allclean              every case, every mechanism
```

A case holds one copy of everything that does not depend on the mechanism. What does —
the initial fields, the species table — sits beside it once per mechanism, and a run picks
one. Nothing is copied between cases: every `constant/<mechanism>` reaches the same
`mechanisms/<mechanism>`.

| Mechanism | Species | Reactions | |
|---|---|---|---|
| [h2o2](mechanisms/h2o2/) | 10 | 29 | H2/O2 submechanism of GRI-Mech 3.0, plus N2 |
| [grimech](mechanisms/grimech/) | 53 | 325 | GRI-Mech 3.0 |
| [C1_C3_HT_NOX_159_2459](mechanisms/C1_C3_HT_NOX_159_2459/) | 159 | 2459 | CRECK C1-C3 high-temperature + NOx |

Each mechanism's README says where it comes from and how to regenerate it.

## The cases

| Case | Flame | Cells | Turbulence | Combustion | Mechanisms |
|---|---|---|---|---|---|
| [2D_planar_flame](2D_planar_flame/) | freely propagating premixed H2/air front, phi = 0.5 | 40 000 | laminar | `laminar` | all three |
| [2D_jet_flame](2D_jet_flame/) | axisymmetric non-premixed jet, pure H2 into air | 40 000 | laminar | `laminar` | all three |
| [3D_nonPremixed_burner](3D_nonPremixed_burner/) | coaxial H2/air injector in a confinement | 39 936 | RAS k-epsilon | `PaSR` | all three |
| [3D_premixed_burner](3D_premixed_burner/) | premixed H2/air, phi = 0.6, through a sudden expansion | 39 924 | RAS k-epsilon | `PaSR` | all three |
| [nonUniformHeating2D](nonUniformHeating2D/) | homogeneous H2/air autoignition, periodic box | 16 384 | laminar | `laminar` | all three |
| [counterFlowFlame2D](counterFlowFlame2D/) | CH4/air counterflow diffusion flame | 32 768 | laminar | `laminar` | grimech, C1_C3 |

`counterFlowFlame2D` burns methane, which h2o2 does not have, so it has no `0.h2o2`. In the
hydrogen cases the carbon species of the larger mechanisms start at zero, through
`Ydefault`, and stay there.

Both combustion models integrate the chemistry in every cell every step — `PaSR` only
scales the source term afterwards — so every case measures the integrator rather than a
switch deciding how often to call it.

The four flames are about 40 000 cells on purpose. `kodesChemistryModel` integrates one
cell per thread, and the device this is benchmarked on carries of order 10 000 threads at
once: a mesh below that leaves most of a batch idle, so the run times launch overhead and
memory traffic rather than the integrator. Forty thousand is a few times the device
width — enough that a batch is full and that the spread of per-cell step counts a real
flame has is present within it. `nonUniformHeating2D` is the opposite: every cell is the
same reactor, so every thread has the same work, and it is the quickest check that a
mechanism and a model run at all.

## Running a case

```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc
cd 2D_planar_flame

./Allrun                                # as the case is set up now (h2o2, standard)
./Allrun grimech                        # another mechanism, the same variant
./Allrun grimech kodes                  # another mechanism and variant
./Allrun h2o2 standard --np 4           # decomposePar, then 4 MPI ranks
./Allclean
```

| Variant | `chemistryType/method` | Needs |
|---|---|---|
| `standard` | `standard` | nothing; this is the reference |
| `kodes` | `kodesChemistryModel` | CUDA device + libraries |

The libraries are
`libkodesChemistryModel.so` and `libkodes_<mechanism>.so`, both built by the top-level
`./Allwmake`.

`./Allrun` is [utilities/kodesCase.py](../utilities/kodesCase.py), which uses foamlib
from the repository's `.venv` (see the top-level README), or from `$KODES4FOAM_PYTHON`.
Setting a case up for a mechanism and a variant is four edits, made together so they
cannot disagree:

| What | Becomes |
|---|---|
| `0/` | a copy of `0.<mechanism>` |
| `constant/thermophysicalProperties` | `foamChemistryFile` and `foamChemistryThermoFile` under `<constant>/<mechanism>/` |
| `constant/chemistryProperties` | `chemistryType/method`; `kodesSetting/mechanism` |
| `system/controlDict` | `libs (libkodesChemistryModel.so)` for kodes, `libs ()` for standard |

`kodesCase.py setup <case> <mechanism> <variant>` makes just those edits, without
running; `kodesCase.py list <case>` says what a case has and what it is set up for now.
A run leaves the case set up for what it ran, so `git diff` shows which that was.

The solver log is `log.reactingFoam.<mechanism>.<variant>`, so the timings of different
runs survive side by side; the fields do not — they are those of the last run.
`kodesCase.py log <log>...` summarises logs: steps, wall clock, wall clock per step
without the first, the first fatal error, and the `kodesChemistryModel` banner.

From this directory, `./Allrun` runs the `standard` variant of every case with every
mechanism it has, `./Allrun kodes` the kodes variant instead, and
`./Allclean` cleans every case.

## Parametric study

[utilities/parametricStudy.ipynb](../utilities/parametricStudy.ipynb) runs one case over
the whole grid — mechanisms × `standard`/`kodes` — each point on its own foamlib clone in
`runs/<case>/<mechanism>.<variant>`, so the tutorial stays as it is and every point keeps
its fields. It tabulates the logs, compares each kodes run's fields with the standard run
of the same mechanism, and plots cost per step and speed-up over `standard`. Points that
already finished are not rerun.

## What a run costs

Measured on one core, `standard` variant. The `grimech` and `C1_C3` columns are the
`h2o2` measurement scaled by the per-step cost of each mechanism relative to `h2o2`
(11.5x for `grimech`, 108x for `C1_C3`), taken from the first twenty steps of
`2D_planar_flame` on each:

| Case | Steps | `h2o2` | `grimech` | `C1_C3` |
|---|---|---|---|---|
| 2D_planar_flame | 543 | 53 s | 10 min | 1.6 h |
| 2D_jet_flame | 179 | 2 min | 24 min | 3.7 h |
| 3D_nonPremixed_burner | 69 | 2 min | 25 min | 3.9 h |
| 3D_premixed_burner | 70 | 3 min | 36 min | 5.6 h |

`endTime` is set for that: these are demonstration lengths — long enough that a flame
exists, burns and can be measured, not long enough to be a converged simulation of
anything. What each `h2o2` run ends at:

| Case | max T | max OH | Reading |
|---|---|---|---|
| 2D_planar_flame | 1651 K | — | front sitting near where it was started, at its own adiabatic temperature |
| 2D_jet_flame | 2220 K | — | diffusion flame well above the 1800 K it was started from |
| 3D_nonPremixed_burner | 1957 K | 0.010 | lit and near stoichiometric (max H2O 0.238); the flame has not reached the outlet |
| 3D_premixed_burner | 1870 K | 0.0055 | burning in the recirculation behind the step; the mixing tube is still full of fresh charge |

Raising `endTime` is the first thing to do with a machine that has more than one core;
`utilities/cylindricalBurnerBlockMesh.py --refine` is the second, for the two 3D meshes
(see [geometry/](geometry/)).

## Parallel

Every case has a `system/decomposeParDict`, `scotch` with a `hierarchical` fallback
commented out beneath it; `./Allrun <mechanism> <variant> --np N` sets
`numberOfSubdomains` to N, decomposes and runs on N ranks. With `gatherCells true`, the
default, the kodes variants still integrate the whole mesh as one ensemble on the master
rank's GPU — see the top-level README.

## What these cases are not

**Unity Lewis number.** ESI `reactingFoam` diffuses every specie at the same rate.
Hydrogen's is roughly three times the mixture's, so the laminar flame speed and the lean
flammability limit these cases produce are not the right ones, and preferential-diffusion
effects — the thing hydrogen flames are interesting for — are absent. It does not affect
what is being measured here: every variant sees the same transport, and the chemistry
integrator is handed the same composition either way.

**Reconstructed 3D geometry.** The two burners are representative laboratory dimensions,
generated from [geometry/](geometry/); see the `_comment` block at the top of each JSON
for which numbers are assumptions. Nothing here should be compared against published data
for a real rig.

**Started, not ignited.** The four flames begin with a region of equilibrium products at
their adiabatic flame temperature, so a flame exists from the first step. That is a
deliberate choice over a hot reactant pocket, which lights the same flame through a
pressure pulse that takes longer to leave the domain than the flame takes to form. It
does mean the first tenth of a millisecond of every run is the initial condition
relaxing, not physics.

**320 K, not 300 K.** Every one of the three mechanisms has species whose JANAF fit
starts at `Tlow = 300` — 92 of the 159 in `C1_C3`. A field sitting exactly on that bound
spends the run a hair below it, because `T` is recovered from `h` by Newton iteration,
and each such cell writes a `janafThermo::limit()` warning per specie per iteration.
Twenty kelvin of margin removes all of them and changes nothing else.

**`C1_C3` with `standard` is not a like-for-like reference.** OpenFOAM cannot read its
236 PLOG reactions, so the `standard` path integrates them without their pressure
dependence; compare cost on it, and results on `h2o2` or `grimech`. See
[its README](mechanisms/C1_C3_HT_NOX_159_2459/README.md), which also covers OpenFOAM's
`seulex` failing on it.
