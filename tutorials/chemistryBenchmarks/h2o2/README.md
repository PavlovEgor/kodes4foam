# h2o2 -- 10 species, 29 reactions

H2/O2 submechanism of GRI-Mech 3.0 plus N2. The only one of the three whose chemistry is entirely about what these cases burn.

| Case | Flame | Cells |
|---|---|---|
| [2D_planar_flame](2D_planar_flame/) | freely propagating premixed front, phi = 0.5 | 40 000 |
| [2D_jet_flame](2D_jet_flame/) | axisymmetric non-premixed jet, pure H2 into air | 40 000 |
| [3D_nonPremixed_burner](3D_nonPremixed_burner/) | coaxial H2/air injector in a confinement | 39 936 |
| [3D_premixed_burner](3D_premixed_burner/) | premixed phi = 0.6 through a sudden expansion | 39 924 |

What the cases are, how the five chemistry variants are switched and what the
whole set is for is in [../README.md](../README.md).

## mech/

| File | Read by |
|---|---|
| `thermo/reactions` | `foamChemistryReader`, via `constant/thermophysicalProperties` |
| `thermo/thermo.compressibleGas` | the same, for JANAF and Sutherland coefficients |
| `mechanism.cuh` | `kodesChemistryModel`, for the pyJac specie ordering |

Each case reaches these through a `constant/mech` symlink, so the four cases of
this set cannot drift onto different copies of the mechanism.

The files themselves are the ones generated for the DLBFoam tutorial sets; how
they were produced from the CHEMKIN sources, and what was checked about them,
is in
[../../DLBFoam-Hydrogen-Tutorials/h2o2/README.md](../../DLBFoam-Hydrogen-Tutorials/h2o2/README.md).

## The GPU library

`./Allwmake` builds `libkodesChemistryModel` and `libkodes_h2o2` into
`$FOAM_USER_LIBBIN` -- the first is what the `kodes-*` variants' `chemistryLibs`
files load, the second what their `kodesSetting` names with `mechanism h2o2;`.
The model is one library for every mechanism; the mechanism is a KODES
library of its own, built with `nvcc`, since pyJac writes it as CUDA source
for the KODES kernels to call into. The `standard` variant needs neither.
