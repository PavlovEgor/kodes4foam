# OpenFOAM Chemistry Model Utilities

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
  tutorials/DLBFoam-Hydrogen-Tutorials/mech/thermo/
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
T=tutorials/DLBFoam-Hydrogen-Tutorials
for m in h2o2 grimech C1_C3_HT_NOX_159_2459; do
  ./utilities/checkKodesMechanism.py \
    $T/$m/mech/mechanism.cuh \
    $T/$m/mech/thermo/reactions \
    --thermophysical-properties $T/$m/2D_planar_flame/constant/thermophysicalProperties
done
```
