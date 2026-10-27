# grimech — 53 species, 325 reactions

GRI-Mech 3.0, the middle of the three mechanisms by size. It is a methane mechanism, but
it contains the full H2/O2 subset and the NOx chemistry, so a hydrogen-air flame is
inside its range of validity; the 38 carbon species simply start at zero and stay there.
It is also the mechanism of `counterFlowFlame2D`, which burns methane.

| File | Read by |
|---|---|
| `reactions` | `foamChemistryReader`, via a case's `constant/thermophysicalProperties` |
| `thermo.compressibleGas` | the same, for JANAF and Sutherland coefficients |
| `mechanism.cuh` | symlink to `pyJacSource/grimech/out/mechanism.cuh`, for [checkKodesMechanism.py](../../../utilities/checkKodesMechanism.py) |

## The mechanism

[`pyJacSource/grimech/grimech30.dat`](../../../src/kodesChemistryModel/pyJacSource/grimech/grimech30.dat)
and `thermo30.dat`, unmodified — the same files the checked-in pyJac output in
`pyJacSource/grimech/out` was generated from.

GRI-Mech does not ship a CHEMKIN transport file here, so the Lennard-Jones data the
Sutherland coefficients are fitted from is taken from Cantera's bundled `gri30.yaml`,
which is the same mechanism. Its species list was checked against the pyJac header
first: the same 53 names, no additions, no omissions.

Every specie gets its own fitted Sutherland pair. (The `thermo.compressibleGasGRI` the
old `counterFlowFlame2D` carried had OpenFOAM's defaults, `As = 1.67212e-06`,
`Ts = 170.672`, on all 53 species — no transport data at all.)

N2 is the last specie in pyJac's ordering and so the one eliminated from the state
vector — the same specie `inertSpecie` names.

## Regenerating

```bash
source .venv/bin/activate
source /usr/lib/openfoam/openfoam2412/etc/bashrc
P=src/kodesChemistryModel/pyJacSource/grimech

# The pyJac CUDA sources in $P/out are already in the repository. To rebuild:
python -m pyjac -l cuda -i $P/grimech30.dat -t $P/thermo30.dat -b $P/out

# 'gri30.yaml' is a Cantera mechanism name, not a file in the repository:
# chemkinToFoam.py takes the transport data straight from it.
./utilities/chemkinToFoam.py $P/grimech30.dat $P/thermo30.dat gri30.yaml \
    tutorials/mechanisms/grimech
```
