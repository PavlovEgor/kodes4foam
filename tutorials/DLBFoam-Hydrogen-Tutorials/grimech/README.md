# grimech — 53 species, 325 reactions

GRI-Mech 3.0, the middle of the three sets by size. It is a methane mechanism, but it
contains the full H2/O2 subset and the NOx chemistry, so a hydrogen-air flame is inside
its range of validity; the 38 carbon species simply start at zero and stay there.

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

Being the mechanism `kodesChemistryModel` was first built against, this set is the
closest thing to a control: if a case here behaves differently from
[counterFlowFlame2D](../../counterFlowFlame2D), the mechanism is not the reason.

## The mechanism

[`pyJacSource/grimech/grimech30.dat`](../../../src/kodesChemistryModel/pyJacSource/grimech/grimech30.dat)
and `thermo30.dat`, unmodified — the same files the checked-in pyJac output in
`pyJacSource/grimech/out` was generated from.

GRI-Mech does not ship a CHEMKIN transport file here, so the Lennard-Jones data the
Sutherland coefficients are fitted from is taken from Cantera's bundled `gri30.yaml`,
which is the same mechanism. Its species list was checked against the pyJac header
first: the same 53 names, no additions, no omissions.

N2 is the last specie in pyJac's ordering and so the one eliminated from the state
vector — the same specie `inertSpecie` names.

## Regenerating

```bash
source venv/bin/activate
source /usr/lib/openfoam/openfoam2412/etc/bashrc
P=src/kodesChemistryModel/pyJacSource/grimech

# The pyJac CUDA sources in $P/out are already in the repository. To rebuild:
python -m pyjac -l cuda -i $P/grimech30.dat -t $P/thermo30.dat -b $P/out

# 'gri30.yaml' is a Cantera mechanism name, not a file in the repository:
# chemkinToFoam.py takes the transport data straight from it.
./utilities/chemkinToFoam.py $P/grimech30.dat $P/thermo30.dat gri30.yaml \
    tutorials/DLBFoam-Hydrogen-Tutorials/grimech/mech/thermo
cp $P/out/mechanism.cuh tutorials/DLBFoam-Hydrogen-Tutorials/grimech/mech/
```

The `thermo.compressibleGas` this produces is not the same as
`counterFlowFlame2D/constant/thermo.compressibleGasGRI`, which has the same
`As = 1.67212e-06`, `Ts = 170.672` on all 53 species — OpenFOAM's defaults, i.e. no
transport data at all. Here every specie gets its own fitted pair.

## The initial condition of 2D_planar_flame

All 15 species the Cantera profiles in `2D_planar_flame/0.orig` cover — H2, H, O, O2,
OH, H2O, HO2, H2O2, N, NH, NH2, NNH, N2O, NO, N2 — are in GRI-Mech, so the initial
condition transfers whole. The remaining 38 species start at `Ydefault`, i.e. zero.

The profiles are a Cantera 1D solution of a lean (phi = 0.5) hydrogen-air flame on the
*Capurso* mechanism, so they are an initial condition rather than a converged solution
of this mechanism; the flame relaxes to GRI-Mech's own structure over the first few
flame times.


The other three cases set their species uniformly from the streams that feed them --
premixed phi = 0.6 hydrogen-air for `3D_AHEAD`, pure H2 against air for `2D_jet_flame`
and `3D_HYLON_flame_A` -- so only H2, O2 and N2 have fields of their own there and every
other specie of the mechanism starts at `Ydefault`, i.e. zero. That is the same in all
three sets, which is why those three cases are byte-identical between sets apart from
the `mechanism` named in `kodesSetting`.

## Status

All four cases run. Verified with the stock `standard` chemistry model on CPU -- this
machine has no GPU and no nvcc, so `libkodes_grimech` was never built and
the KODES path itself is unverified. What was checked offline instead is the specie
mapping, via [checkKodesMechanism.py](../../../utilities/checkKodesMechanism.py).

`2D_planar_flame` was run end to end: 53 species and 325 reactions selected, the flame
holding at `min/max(T) = 298, 1648 K`. The other three cases are byte-identical to the
`h2o2` set's apart from the `mechanism` named in `kodesSetting` and the mechanism the `mech`
symlink reaches, and those were run there -- see [../h2o2/README.md](../h2o2/README.md).
