# OpenFOAM Chemistry Model Utilities

The Python ones run from the repository's `.venv`:
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## kodesCase.py

Sets a tutorial case up for one mechanism and one chemistry variant and runs it, with
[foamlib](https://github.com/gerlero/foamlib). Every case's `./Allrun` is
`kodesCase.py run . "$@"`.

A case has a `0.<mechanism>` and a `constant/<mechanism>` symlink for each mechanism it
can run with. Setting it up copies `0.<mechanism>` to `0`, points
`foamChemistryFile`/`foamChemistryThermoFile` at `constant/<mechanism>`, sets
`chemistryType/method` and `kodesSetting` (`mechanism`, `balanceBatches`, `balancerKey`),
and `libs` in `controlDict` — see [../tutorials/README.md](../tutorials/README.md).

```bash
./kodesCase.py list  <case>                         # mechanisms, variants, current set-up
./kodesCase.py setup <case> <mechanism> <variant>   # set up only
./kodesCase.py run   <case> [mechanism] [variant] [--np N]
./kodesCase.py log   <log>...                       # steps, wall clock, per-step cost, errors
```

Variants: `standard`, `kodes-none`, `kodes-temperature`, `kodes-rhsNorm`,
`kodes-stiffness`. The same functions are a module — `mechanisms`, `configure`, `clone`,
`run`, `read_log`, `compare` — which is how the notebook below drives the cases. `clone`
is foamlib's, plus re-pointing the `constant/<mechanism>` symlinks, which are relative to
the tutorial and would reach nothing from a clone elsewhere.

## parametricStudy.ipynb

One case over the grid of mechanisms × `standard`/`kodes` × the four KODES balancers.
Each point is a `kodesCase.clone` in `runs/<case>/<mechanism>.<variant>` (ignored by
git), run with `kodesCase.run`; the notebook then tabulates the logs, compares each kodes
run's fields with the standard run of the same mechanism, and plots cost per step,
speed-up over `standard` and what balancing gains over `kodes-none`. Points that already
finished are skipped, so re-running after adding a mechanism or a balancer only runs the
new points. The case, the grid and `controlDict` overrides (a shorter `endTime` for a
first pass) are set in its parameters cell.

## chemkinToFoam.py

Converts a CHEMKIN-III mechanism (kinetics + thermo + transport) to OpenFOAM format by
driving the real, upstream `chemkinToFoam` utility instead of re-parsing CHEMKIN syntax
by hand. Produces correct reaction types (reversible / third-body / Troe / Lindemann
fall-off), real per-specie NASA thermo data, and real per-specie Sutherland transport
coefficients (`As`, `Ts`) fitted from the mechanism's own Lennard-Jones transport data via
Cantera.

Four things about real-world CHEMKIN files that trip up OpenFOAM's `chemkinToFoam` are
handled automatically:

- **PLOG** (pressure-dependent Arrhenius) auxiliary lines aren't supported by OpenFOAM's
  chemkinReader at all ("unknown third-body specie PLOG"). They're stripped, so those
  reactions fall back to their own leading (high-pressure-limit) Arrhenius parameters —
  pressure-dependence is lost for the affected reactions.
- The `THERMO` keyword of a standalone thermodynamics file is forced to `THERMO ALL`.
  A bare `THERMO` puts the reader into the mode it uses for a thermo section embedded in
  the kinetics file, where no global temperature-range line is expected — so it reads
  that line as the first specie record and fails on its element columns.
- The temperature-range line itself must be an exact fixed-width 3×10 column record
  (Fortran `3F10.0`); free-format thermo files are reformatted to match.
- Element symbols in the `ELEMENTS` block are upper-cased. OpenFOAM only accepts
  upper-case element names, so the mixed-case symbols Cantera's `yaml2ck` writes (`Ar`,
  `He`) are rejected; CHEMKIN itself is case-insensitive here.

`chemkinToFoam` also doesn't read a CHEMKIN `.TRAN` transport file directly — its
`<CHEMKINTransport>` argument must already be an OpenFOAM dictionary of per-specie
`sutherlandTransport` coefficients, which this script computes.

### Requirements

- `cantera` (Python package) — used for mechanism parsing and transport property
  evaluation.
- The OpenFOAM `chemkinToFoam` binary on `$PATH` (source an OpenFOAM environment first).

### Usage

```bash
./chemkinToFoam.py <kinetics.CKI> <thermo.CKT> <transport> <output_dir> \
  [--chemkinToFoam PATH] [--n-fit-points N]
```

`<transport>` is where the Lennard-Jones data for the Sutherland fit comes from, and can
be either:

- a CHEMKIN `.TRAN` file, combined with `<kinetics.CKI>` and `<thermo.CKT>`; or
- a ready Cantera mechanism whose name ends in `.yaml` — including a bundled one such as
  `gri30.yaml`, which is how the `grimech` set is generated, GRI-Mech shipping no `.TRAN`
  of its own here.

### Example

```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc
./utilities/chemkinToFoam.py \
  external/Kinetic-Mechanisms/Gas-Phase/CoreMechanism_C0-C4/Soot-NOx/C1_C3_HT_NOX_159_2459/kinetics.CHEMKIN.CKI \
  external/Kinetic-Mechanisms/Gas-Phase/CoreMechanism_C0-C4/Soot-NOx/C1_C3_HT_NOX_159_2459/thermo.CHEMKIN.CKT \
  external/Kinetic-Mechanisms/Gas-Phase/CoreMechanism_C0-C4/Soot-NOx/C1_C3_HT_NOX_159_2459/TOT2003.TRAN \
  tutorials/mechanisms/C1_C3_HT_NOX_159_2459/
```

## checkKodesMechanism.py

Checks that a case's OpenFOAM species table and its pyJac mechanism header describe the
same set of species — that is, that `Foam::kodesSpecieMap` will build rather than call
`FatalError` when the case starts.

That check normally needs a running case and therefore `libkodes_<mechanism>.so`, built
with nvcc. This performs the identical check offline, from the two files
alone, so a case can be validated on a machine with no GPU. It reads the specie names
from the `Species Indexes` block of `mechanism.cuh` (cross-checking them against `NSP`)
and from the `species (...)` list of the OpenFOAM reactions file, then requires the two
to be the same size, the same set, and free of duplicates, and reports which specie
pyJac eliminates from the state vector.

### Usage

```bash
./checkKodesMechanism.py <mechanism.cuh> <reactions> \
  [--thermophysical-properties <thermophysicalProperties>]
```

With `--thermophysical-properties` it also compares the case's `inertSpecie` against
that eliminated specie. The two are independent settings — OpenFOAM's `inertSpecie` only
balances the Y equations — but pointing both at the same, most abundant, specie keeps
round-off down, so a mismatch is reported as a warning rather than a failure.

### Example

```bash
T=tutorials
for m in h2o2 grimech C1_C3_HT_NOX_159_2459; do
  ./utilities/checkKodesMechanism.py \
    $T/mechanisms/$m/mechanism.cuh \
    $T/mechanisms/$m/reactions \
    --thermophysical-properties $T/2D_planar_flame/constant/thermophysicalProperties
done
```
