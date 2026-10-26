# kodes4foam

`kodesChemistryModel`: an OpenFOAM v2412 chemistry model that integrates the cell-wise
chemistry on the GPU with [KODES](https://github.com/PavlovEgor/KODES), one cell per
thread, in batches.

The model itself is plain OpenFOAM C++. KODES and the mechanism are a separate shared
library per mechanism, `libkodes_<mechanism>.so`, which the model opens when a case
starts. So one `libkodesChemistryModel.so` serves every mechanism, and moving a case to
another mechanism is a dictionary edit, not a rebuild.

## Layout

```
external/KODES/                 the KODES submodule, 0.23.1
src/kodesChemistryModel/
  kodesChemistryModel.{C,H}     the model
  kodesSpecieMap.{C,H}          KODES/pyJac <-> OpenFOAM specie ordering
  makeKodesChemistry*Types.*    runtime selection for psi/rhoReactionThermo
  pyJacSource/<mechanism>/out/  pyJac's CUDA output, one directory per mechanism
  Allwmake                      builds the model and the KODES libraries
Allwmake                        the same, from the top
```

## Building

Needs OpenFOAM v2412 sourced; for the KODES libraries also CMake ≥ 3.18 and nvcc.

```bash
git submodule update --init external/KODES

./Allwmake                      # the model + libkodes_<m>.so for every pyJacSource/<m>
./Allwmake grimech              # the model + libkodes_grimech.so only
```

Everything goes to `$FOAM_USER_LIBBIN`. `NVARCH` sets the compute capability the
KODES libraries are built for (default `120`, an RTX 50xx):
`NVARCH=80 ./Allwmake` for an A100.

The model alone builds with `wmake libso src/kodesChemistryModel` and needs no CUDA.

## Using it in a case

`system/controlDict`:

```
libs            ( libkodesChemistryModel.so );
```

`constant/chemistryProperties`:

```
chemistryType
{
    solver          ode;
    method          kodesChemistryModel;
}

chemistry       on;

initialChemicalTimeStep 1e-7;

// absTol, relTol and maxSteps are passed on to KODES
odeCoeffs
{
    solver          seulex;
    absTol          1e-12;
    relTol          1e-3;
}

kodesSetting
{
    mechanism         grimech;      // opens libkodes_grimech.so
    integrationMethod seulex;       // seulex | euler
    deviceShare       best;         // best | half
    balanceBatches    true;
    balancerKey       temperature;  // temperature | rhsNorm | stiffness | none
}
```

The full list of `kodesSetting` entries is in the header of
[kodesChemistryModel.H](src/kodesChemistryModel/kodesChemistryModel.H). The species of
the library are matched by name against the case's species table when the case starts;
a library built for another mechanism stops the run there, naming the specie it lacks.

## Adding a mechanism

Generate pyJac's CUDA output into `src/kodesChemistryModel/pyJacSource/<name>/out`
(`python -m pyjac -l cuda ... -b <that directory>`) and run `./Allwmake <name>`. Cases
then select it with `mechanism <name>;`.

Mind the names: KODES's own `data/mechanisms/h2o2` (9 species) and `pyJacSource/h2o2`
here (10, with N2) both build `libkodes_h2o2.so`. Installing KODES's default build into
`$FOAM_USER_LIBBIN` would replace this one.

## License

GNU General Public License v3, as OpenFOAM.
