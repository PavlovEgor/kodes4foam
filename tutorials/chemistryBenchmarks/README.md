# chemistryBenchmarks

Three detailed mechanisms, four `reactingFoam` hydrogen flames each, and five
chemistry-model settings per flame. The point is to be able to time
[`kodesChemistryModel`](../../src/kodesChemistryModel) -- the GPU chemistry model
and each of its batch balancers -- against the chemistry model OpenFOAM ships,
on the same flame, the same mesh and the same tolerances, so that a difference
in wall-clock is a difference in the integrator and nothing else.

Everything here is OpenFOAM v2412 and stock `reactingFoam`. Nothing but the
chemistry model changes between variants: same mesh, same initial and boundary
conditions, same `odeCoeffs`, same combustion model.

## Layout

```
chemistryBenchmarks/
  geometry/                     the two 3D burner geometries, as JSON
  <mechanism>/
    mech/                       reactions, thermo and the pyJac header
    Allwmake                    builds the model and libkodes_<mechanism>
    <case>/
      chemistrySetups/          the five variants, one directory each
      setChemistry              puts one of them in place
      Allrun                    ./Allrun [variant]
      constant/mech             symlink to ../../mech
```

The four cases are the same four in every mechanism set; only `constant/mech`
and the `mechanism` in `chemistrySetups/kodes-*/chemistryProperties` differ.

## The four cases

| Case | Flame | Cells | Turbulence | Combustion | `endTime` |
|---|---|---|---|---|---|
| [2D_planar_flame](h2o2/2D_planar_flame/) | freely propagating premixed front, phi = 0.5 | 40 000 | laminar | `laminar` | 0.5 ms |
| [2D_jet_flame](h2o2/2D_jet_flame/) | axisymmetric non-premixed jet, pure H2 into air | 40 000 | laminar | `laminar` | 1 ms |
| [3D_nonPremixed_burner](h2o2/3D_nonPremixed_burner/) | coaxial H2/air injector in a confinement | 39 936 | RAS k-epsilon | `PaSR` | 0.5 ms |
| [3D_premixed_burner](h2o2/3D_premixed_burner/) | premixed phi = 0.6 through a sudden expansion | 39 924 | RAS k-epsilon | `PaSR` | 0.5 ms |

Both combustion models integrate the chemistry in every cell every step --
`PaSR` only scales the source term afterwards -- so all four cases measure the
integrator rather than a switch that decides how often to call it.

All four are about 40 000 cells, on purpose and not because the physics asked
for that number. `kodesChemistryModel` integrates one cell per thread, and the
device this is benchmarked on carries of order 10 000 threads at once: a mesh
below that leaves most of a batch idle, so the run times launch overhead and
memory traffic rather than the integrator, and a balancer has almost nothing to
reorder. Forty thousand is a few times the device width -- enough that a batch
is full, that the spread of per-cell step counts a balancer exists to exploit
is actually present within it, and that the three balancers can be told apart.

That the CPU reference is then slow is the price; see below.

## The five variants

| Variant | `chemistryType/method` | `balancerKey` | Needs |
|---|---|---|---|
| `standard` | `standard` | -- | nothing; this is the reference |
| `kodes-none` | `kodesChemistryModel` | `balanceBatches false` | CUDA device + library |
| `kodes-temperature` | `kodesChemistryModel` | `temperature` | CUDA device + library |
| `kodes-rhsNorm` | `kodesChemistryModel` | `rhsNorm` | CUDA device + library |
| `kodes-stiffness` | `kodesChemistryModel` | `stiffness` | CUDA device + library |

`kodes-none` is the measurement the three balanced runs are compared against:
it isolates what the GPU port is worth before any load balancing.

Each variant directory holds two files, and `./setChemistry` moves both:

* `chemistryProperties` -- which model, and how it is configured
* `chemistryLibs` -- the `libs` entry, `#include`d by `system/controlDict`

They travel together on purpose. `chemistryProperties` names the model and
`chemistryLibs` loads the library that provides it; a case with one and not the
other either stops at *Unknown chemistryModel type kodesChemistryModel* or
loads a CUDA library it never calls. Because `controlDict` includes the file
rather than containing the entry, switching variants never edits `controlDict`.

## Running

```bash
cd h2o2/2D_planar_flame
./Allrun standard              # blockMesh, setFields, reactingFoam
./Allrun kodes-temperature     # same, with the GPU model
```

`./Allrun` with no argument runs whatever variant is currently in place.
`./setChemistry <variant>` switches without running. The log is named after the
variant (`log.reactingFoam.standard`), so two variants' timings survive side by
side; the time directories are not, so copy the case if you want to keep two
sets of fields.

From this directory, `./Allrun` runs the `standard` variant of every case of
every mechanism, and `./Allclean` cleans them all.

## What a run costs

Measured on one core of this machine, `standard` variant. The right-hand
columns are that measurement scaled by the per-step cost of each mechanism
relative to `h2o2` (11.5x for `grimech`, 108x for `C1_C3`), taken from the
first twenty steps of `2D_planar_flame` on each:

| Case | Steps | `h2o2` | `grimech` | `C1_C3` |
|---|---|---|---|---|
| 2D_planar_flame | 543 | 53 s | 10 min | 1.6 h |
| 2D_jet_flame | 179 | 2 min | 24 min | 3.7 h |
| 3D_nonPremixed_burner | 69 | 2 min | 25 min | 3.9 h |
| 3D_premixed_burner | 70 | 3 min | 36 min | 5.6 h |

Per cell per step, the whole solver -- transport, pressure and chemistry
together, of which chemistry is nearly all of it:

| Mechanism | Species | Reactions | Per cell per step |
|---|---|---|---|
| `h2o2` | 10 | 29 | 60 us |
| `grimech` | 53 | 325 | 690 us |
| `C1_C3_HT_NOX_159_2459` | 159 | 2459 | 6.5 ms |

`endTime` is set for that: these are demonstration lengths -- long enough that
a flame exists, burns and can be measured, not long enough to be a converged
simulation of anything. What each `h2o2` run ends at:

| Case | max T | max OH | Reading |
|---|---|---|---|
| 2D_planar_flame | 1651 K | -- | front sitting near where it was started, at its own adiabatic temperature |
| 2D_jet_flame | 2220 K | -- | diffusion flame well above the 1800 K it was started from |
| 3D_nonPremixed_burner | 1957 K | 0.010 | lit and near stoichiometric (max H2O 0.238); the flame has not reached the outlet |
| 3D_premixed_burner | 1870 K | 0.0055 | burning in the recirculation behind the step; the mixing tube is still full of fresh charge |

Raising `endTime` is the first thing to do with a machine that has more than
one core; `utilities/cylindricalBurnerBlockMesh.py --refine` is the second, for
the two 3D meshes.

## Parallel

Every case ships a `system/decomposeParDict` it does not use. The runs are
configured serial because the machine these were built on has no GPU and eight
cores; `decomposePar && mpirun -np N reactingFoam -parallel` needs no other
edit. `scotch` is the method, with a `hierarchical` fallback commented out
beneath it.

## What these cases are not

**Unity Lewis number.** ESI `reactingFoam` diffuses every specie at the same
rate. Hydrogen's is roughly three times the mixture's, so the laminar flame
speed and the lean flammability limit these cases produce are not the right
ones, and preferential-diffusion effects -- the thing hydrogen flames are
interesting for -- are absent. The upstream tutorials this work started from
get that right with FickianTransportFoam's `mixtureAveraged` model, which is a
Foundation-branch library and has no v2412 equivalent. It does not affect what
is being measured here: every variant sees the same transport, and the
chemistry integrator is handed the same composition either way.

**Reconstructed 3D geometry.** The two burners are representative laboratory
dimensions, generated from [geometry/](geometry/); see the `_comment` block at
the top of each JSON for which numbers are assumptions. They are not the HYLON
or AHEAD rigs, and nothing here should be compared against published data for
those.

**Started, not ignited.** All four cases begin with a region of equilibrium
products at their adiabatic flame temperature, so a flame exists from the first
step. That is a deliberate choice over a hot reactant pocket, which lights the
same flame through a pressure pulse that takes longer to leave the domain than
the flame takes to form. It does mean the first tenth of a millisecond of every
run is the initial condition relaxing, not physics.

**320 K, not 300 K.** Every one of the three mechanisms has species whose JANAF
fit starts at `Tlow = 300` -- 92 of the 159 in `C1_C3`. A field sitting exactly
on that bound spends the run a hair below it, because `T` is recovered from `h`
by Newton iteration, and each such cell writes a `janafThermo::limit()` warning
per specie per iteration. The first test run of `2D_planar_flame` produced half
a million of them. Twenty kelvin of margin removes all of them and changes
nothing else.
