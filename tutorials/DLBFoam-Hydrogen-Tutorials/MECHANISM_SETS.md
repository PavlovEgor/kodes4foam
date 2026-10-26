# DLBFoam hydrogen tutorial, ported to the three pyJac mechanisms in this repo

The upstream tutorials in this directory run on the hydrogen/ammonia mechanism of
Capurso et al. Only its pyJac **C** output survives here, in [mech/src](mech/src) —
there is no CHEMKIN input to regenerate it from, and `kodesChemistryModel` needs the
**CUDA** variant, so that mechanism cannot be used with KODES.

The three mechanisms that *do* have usable sources live in
[src/kodesChemistryModel/pyJacSource](../../src/kodesChemistryModel/pyJacSource). This
directory holds one set of cases per mechanism:

| Set | Species | Reactions | KODES library |
|---|---|---|---|
| [h2o2](h2o2/) | 10 | 29 | `libkodes_h2o2` |
| [grimech](grimech/) | 53 | 325 | `libkodes_grimech` |
| [C1_C3_HT_NOX_159_2459](C1_C3_HT_NOX_159_2459/) | 159 | 2459 | `libkodes_C1_C3_HT_NOX_159_2459` |

Each set contains all four of the upstream cases, so the three sets differ only in the
mechanism, which is the point: they span an order of magnitude in species count and two
in reaction count, on identical geometry, identical meshes and identical initial
conditions.

| Case | What it is | Cells | Solver |
|---|---|---|---|
| `2D_planar_flame` | laminar freely propagating premixed flame | 10 000 | `reactingFoam` |
| `2D_jet_flame` | axisymmetric laminar non-premixed jet flame, with a conjugate nozzle | 249 100 | `chtMultiRegionFoam` |
| `3D_HYLON_flame_A` | turbulent non-premixed coaxial swirl flame (LES) | 156 000 | `reactingFoam` |
| `3D_AHEAD` | turbulent premixed swirl flame (LES) | 207 616 | `reactingFoam` |

Two of those needed a mesh built from scratch, and one of them a geometry along with it
— see [Meshes that had to be reconstructed](#meshes-that-had-to-be-reconstructed) below,
which is the part to read before drawing any conclusion about the flames themselves.
The cell counts are the checked-in defaults and are meant to be turned up; the 3D ones
are regenerated from a parameter file with a single `--refine` argument.

## One model, one KODES library per mechanism

`libkodesChemistryModel` is one library for every set. The mechanism is a library of
its own, `libkodes_<mechanism>.so`: KODES and the pyJac output device-linked together,
since pyJac writes the specie count, the rates and the analytical Jacobian as CUDA
source that the KODES kernels call into. The model opens it with `dlopen()` when a
case starts, by the name `mechanism` gives in `kodesSetting`, so moving a case to
another mechanism needs no build.

Each set's `Allwmake` runs the repository's top-level `Allwmake` with
`KODES_MECHANISMS=<mechanism>`, which builds the model and that mechanism's library from
`src/kodesChemistryModel/pyJacSource/<mechanism>/out` in place; the top-level `Allwmake`
alone builds all three. Building one set does not overwrite anything another set needs,
so they can be built in any order.

## Running a set

```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc

cd tutorials/DLBFoam-Hydrogen-Tutorials/h2o2
./Allwmake                  # needs nvcc, once per set

cd 2D_planar_flame       && ./Allrun && cd ..
cd 2D_jet_flame          && ./Allrun && cd ..
cd 3D_HYLON_flame_A      && ./Allrun && cd ..
cd 3D_AHEAD              && ./Allrun && cd ..
```

To run the stock OpenFOAM chemistry instead, for a reference to compare against, swap
the two `method` lines in `constant/chemistryProperties` and comment out the `libs`
entry in `system/controlDict`. Nothing else changes: both models read the same
mechanism through the same species table.

### `CUDA error ...: invalid program counter`

```
PIMPLE: iteration 1
CUDA error at ../../external/KODES/src/Resources/Operator.cu:69: invalid program counter
```

Unresolved. `Operator.cu:69` is the first `cudaMemcpy` after `solveKernel`, and a
`cudaMemcpy` is the first *synchronising* call after it — the launch reports only
launch errors, so whatever the kernel did wrong is reported here. The fault is on
the kernel side; the line number says nothing about where.

It was first read as a device stack overflow, on the grounds that `solveKernel`
reaches the method and the mechanism through virtual calls and an indirect call is
where ptxas stops adding up frames. That reading is wrong: `cuobjdump -res-usage`
on the built library shows ptxas did compute a frame,

```
Function _ZN5kodes11solveKernelE...:
  REG:255 STACK:15600 SHARED:1024 LOCAL:0 CONSTANT[0]:1016
```

and raising `deviceStackSize` to 128 KiB — eight times that — changed nothing but
the size of the launch. So the stack is not what is short.

Note also what `-res-usage` does *not* say. Only `__global__` entry points carry
resource figures; every `__device__` function in the image, `eval_jacob` and its
262 split pieces included, prints `REG:0 STACK:0`. That is the format, not a
finding — it is no evidence about whether those bodies are there.

h2o2 and grimech run; only C1_C3 fails, which finally makes the mechanism the
one variable. Two readings are already ruled out. Nothing is missing from the
device image — all 282 generated sources were in the build, and all 262
`eval_jacob_N` are in the fatbin — and the split into `jacobs/` is not itself
the trigger, since grimech is split too and runs.

What is worth trying next, in descending order of what it settles:

- **`compute-sanitizer --tool memcheck reactingFoam`.** It names the kernel and
  the instruction and separates a bad address from a bad jump. Everything below
  is guessing by comparison; this is not.
- **`integrationMethod euler`.** Euler only ever calls `derivatives()`, never
  `jacobian()`. The Jacobian path holds everything that grows as NSP² —
  `dfdy_` and `a_` in `SeulexDeviceResources`, `jac` in `mechanism_memory`, 159²
  against grimech's 53² — so if euler gets through, the search is down to
  `eval_jacob` and `LUDecompose`.
- **`balancerKey none`.** `rhsNorm` and `stiffness` evaluate the right hand side
  inside `fillKeysKernel`, a second kernel with a second indirect call site.
  Note that `balanceBatches false` is *not* the way to do this on every tree —
  check the banner actually says `Batch balancing = off`.
- **`cuobjdump -res-usage` on the h2o2 library, compared with the C1_C3 one.**
  `solveKernel` takes its sizes at run time and holds no mechanism code, so the
  two `REG:`/`STACK:` lines have to match exactly unless the compiler is
  devirtualising the call and inlining the mechanism into the kernel. C1_C3
  comes out at `REG:255` — the ceiling, past which everything spills — so a
  smaller h2o2 line would say the kernel is a different kernel per mechanism,
  and that is where to look next.
- **One block:** `threadsPerBlock 64; concurrentSystems 64; batchSize 64;`. Still
  dying means sizes and allocations are not the trigger.

## What each set is made of

```
<mechanism>/
├── Allwmake                        build the model and libkodes_<mechanism>
├── README.md                       the mechanism's provenance and caveats
├── mech/
│   ├── mechanism.cuh               copy of the pyJac header, for checkKodesMechanism.py
│   └── thermo/
│       ├── reactions               OpenFOAM species table + kinetics
│       └── thermo.compressibleGas  NASA polynomials + fitted Sutherland coefficients
├── 2D_planar_flame/
│   ├── 0.orig/                     Cantera 1D flame profiles, mapped by codeStream
│   ├── constant/  (mech -> ../../mech)
│   └── system/
├── 2D_jet_flame/                   two regions: gas and the solid nozzle
│   ├── 0.orig/{gas,nozzle}/        (gas species written by Allrun from Yi.mytemplate)
│   ├── constant/{gas,nozzle}/      (gas/mech -> ../../../mech)
│   └── system/{,gas,nozzle}/
├── 3D_HYLON_flame_A/
└── 3D_AHEAD/
    ├── 0.orig/                     uniform; the streams enter through the inlets
    ├── constant/  (mech -> ../../mech)
    └── system/blockMeshDict        generated, see ../geometry/
```

Every case names its mechanism twice: `foamChemistryFile` in `constant`, reached
through the `mech` symlink, and `mechanism` in `kodesSetting`, which picks the KODES
library. Nothing else in a case has to change when it is copied between sets.

At start-up `kodesSpecieMap` matches the species the KODES library reports — read out
of the "Species Indexes" block of this same `mechanism.cuh` when the library was built
— by name against the OpenFOAM species table, and calls FatalError if the two are not
the same set. That check needs the library, and so nvcc, to run, so
[utilities/checkKodesMechanism.py](../../utilities/checkKodesMechanism.py) performs the
identical check offline:

```bash
./utilities/checkKodesMechanism.py \
    tutorials/DLBFoam-Hydrogen-Tutorials/h2o2/mech/mechanism.cuh \
    tutorials/DLBFoam-Hydrogen-Tutorials/h2o2/mech/thermo/reactions \
    --thermophysical-properties \
    tutorials/DLBFoam-Hydrogen-Tutorials/h2o2/2D_planar_flame/constant/thermophysicalProperties
```

All three sets pass, and in all three the specie pyJac eliminates from the state
vector is N2, the same specie as `inertSpecie`.

## How the mechanism files were generated

`reactions` and `thermo.compressibleGas` come from
[utilities/chemkinToFoam.py](../../utilities/chemkinToFoam.py), which drives the real
`chemkinToFoam` and fits per-specie Sutherland coefficients from the mechanism's own
Lennard-Jones data with Cantera. The exact command for each set is in its README.

`mechanism.cuh` is copied from `pyJacSource/<mechanism>/out/`, and the reaction counts
agree with the OpenFOAM files in every set — 29, 325 and 2459 — which is the coarsest
check that the two halves describe the same mechanism.

## Differences from the upstream tutorial

The upstream cases target OpenFOAM 10 (Foundation) with DLBFoam and
FickianTransportFoam. This repository builds against OpenFOAM v2412 (ESI), so:

- **Detailed diffusion is gone.** `constant/thermophysicalTransport` selected
  FickianTransportFoam's `mixtureAveraged` model in the 2D cases and a species-specific
  constant-Lewis-number model in `3D_AHEAD`, and ESI has an equivalent of neither.
  Species diffusion is therefore unity-Schmidt everywhere here. This matters for the
  *physics* the upstream cases were written to show — the cellular thermodiffusive
  instability of a lean hydrogen flame comes from H2 having Le < 1, and the whole point
  of the upstream `LewisNumbersDict` is that it does not — and those structures will not
  develop. It does not affect what these sets are for: the chemistry integration is
  unchanged, and it is what dominates the cost.
- `constant/momentumTransport` is dropped; ESI reads `constant/turbulenceProperties`,
  which the upstream cases already carried alongside it. The LES model itself is
  unchanged: WALE with a `cubeRootVol` filter width, as upstream, and `PaSR` with
  `Cmix = 1` for the subgrid–chemistry interaction.
- `system/fvSolution` gained the `relTol` that ESI requires in the `"Yi.*"` solver
  block.
- DLBFoam's own entries — `loadbalancing`, `refmapping`, `mixtureFractionProperties` —
  are dropped. `kodesChemistryModel` does its own balancing on the device, configured
  by `balanceBatches` and `balancerKey` in `kodesSetting`.
- The `libs` entry names `libkodesChemistryModel`, and `mechanism` in `kodesSetting`
  the KODES library, as above.

## The original 2D_planar_flame

[2D_planar_flame](2D_planar_flame/) is left as it is — it is what the three sets were
copied from, and its README is still the description of the case. It is not runnable in
its own right: its `chemistryProperties` selects `kodesChemistryModel` but puts
`kodesMechanism` at the top level rather than inside the `kodesSetting` sub-dictionary
the model reads, and points it at a `constant/mechanism.cuh` that does not exist. Its
`constant/mech` symlink also reaches the upstream [mech](mech/) directory, whose
`thermo/` now holds the C1_C3 mechanism while its `src/` still holds Capurso's — the two
do not match. Use one of the three sets instead.

## Meshes that had to be reconstructed

`2D_planar_flame` and `2D_jet_flame` are ports: the upstream cases carry their own
`blockMeshDict`, so the mesh in each set is the published one, cell for cell.

The two 3D cases are not ports. Neither ships a mesh or a `0.orig`; their fields and
meshes are the external dataset the top-level [README](README.md) links to, and nothing
could be run from what is in the repository. Their meshes here are **reconstructions**,
built by [utilities/cylindricalBurnerBlockMesh.py](../../utilities/cylindricalBurnerBlockMesh.py)
from the parameter files in [geometry/](geometry/), and they are not the geometry the
papers used. Read those two JSON files before comparing anything to a published number:
each one states, up front, what is taken from the case README and what is assumed.

The two differ in how much is assumed:

- **3D_AHEAD** has real dimensions to work from. The case README gives the mixing
  chamber as 34 mm across, and `system/sampleDict` fixes the chamber size and the
  measurement heights. What is not resolved is the injector — 8 radial slits, 12
  dilution holes and 16 fuel ports of 0.8 mm — which would need the CAD and
  snappyHexMesh. Since the stream is premixed by the mixing-chamber exit, which is what
  the case is about, the injector is represented by that exit condition: a swirled
  premixed inlet at phi = 0.6.
- **3D_HYLON_flame_A** has none. Its README gives no dimensions at all, only a citation.
  The geometry here is a representative laboratory coaxial hydrogen swirl injector —
  6 mm fuel bore, 20 mm air annulus, 100 mm confinement — and the domain starts at the
  injector exit plane, with the two streams entering through concentric patches. It is
  the right *kind* of burner, not that burner.

Both are genuine 3D reacting LES of a swirl-stabilised hydrogen flame, which is what
makes them useful for measuring chemistry throughput. Neither is a validation case.
Their upstream `system/` sampling dictionaries were dropped rather than carried over —
they list the measurement stations of rigs that are not meshed here — and replaced by
radial profiles that match the reconstruction. The originals are still in
[3D_AHEAD/system](3D_AHEAD/system) and
[3D_HYLON_flame_A/system](3D_HYLON_flame_A/system).

To change a geometry or its resolution, edit the JSON and regenerate:

```bash
./utilities/cylindricalBurnerBlockMesh.py \
    tutorials/DLBFoam-Hydrogen-Tutorials/geometry/AHEAD.json \
    tutorials/DLBFoam-Hydrogen-Tutorials/h2o2/3D_AHEAD/system/blockMeshDict \
    --refine 2                      # roughly eight times the cells
```

## The 2D_jet_flame port

The upstream case is `chtMultiRegionFoam` with a solid nozzle region, and that survived
the port intact: ESI's `chtMultiRegionFoam` does do combustion — it builds its fluid
regions on `rhoReactionThermo` — and `kodesChemistryModel` is registered for that thermo
as well as for `psiReactionThermo`. So the conjugate heat transfer into the steel nozzle
is still there. What changed:

- `hePsiThermo` became `heRhoThermo`, which is what the ESI solver's fluid regions use.
- `compressible::turbulentTemperatureCoupledBaffleMixed` became ESI's
  `compressible::turbulentTemperatureRadCoupledMixed`, plus the `kappaMethod` it wants.
- `p_rgh` became absolute rather than gauge. The upstream case carried it relative to a
  `constant/gas/pRef` of 101 kPa; ESI's solver has no `pRef` and forces
  `p_rgh = p - rho*gh` at start-up, so a gauge field there leaves the outlet's `p0`
  101 kPa too low and the case blows up on the first step.
- The nozzle's tabulated `kappa(T)` and `Cv(T)` became constants — see the comment in
  `constant/nozzle/thermophysicalProperties` for what that costs.
- The `heatSource` fvModel that ignites the jet became a `scalarSemiImplicitSource`
  fvOption, whose `fv::option` base already provides the `timeStart`/`duration` pulse.
- The inlet lost its `laminarBL` profile, which ESI's `flowRateInletVelocity` does not
  offer. It is not needed: the inlet is 30 diameters upstream of the nozzle exit and the
  entry length at Re = 330 is about 20.
- The ambient is 301 K rather than 300 K. Several species of every mechanism here have
  their JANAF fit starting at exactly 300 K, and the enthalpy inversion undershoots that
  by round-off; OpenFOAM clamps and carries on, but prints an ungated warning each time.
  At 300 K that buried one smoke test under a 214 MB log. One Kelvin of headroom is well
  inside the uncertainty on "room temperature" and brought the same log to 576 kB.

## No solver of our own

Every case names a stock solver — `reactingFoam` for three of them,
`chtMultiRegionFoam` for `2D_jet_flame`. `kodesChemistryModel` plugs in underneath any
reacting solver as a `chemistryModel`, so the `libs` entry is the whole of the wiring.
It is registered for both `psiReactionThermo` and `rhoReactionThermo`, which is what
lets the same model serve `reactingFoam` and the fluid regions of `chtMultiRegionFoam`.
