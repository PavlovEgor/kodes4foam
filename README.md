# kodes4foam

`kodesChemistryModel`: an OpenFOAM v2412 chemistry model that integrates the cell-wise
chemistry on the GPU with [KODES](https://github.com/PavlovEgor/KODES), one cell per
thread, in batches.

The model itself is plain OpenFOAM C++. KODES and the mechanism are a separate shared
library per mechanism, `libkodes_<mechanism>.so`, which the model opens when a case
starts. So one `libkodesChemistryModel.so` serves every mechanism, and moving a case to
another mechanism is a dictionary edit, not a rebuild.

## Layout

Laid out like an OpenFOAM plugin or module (see `plugins/README.md` of an OpenFOAM
release), so it builds on its own or from inside `$WM_PROJECT_DIR/plugins/`:

```
Allwmake, Allwclean             the plugin entry points
src/
  Allwmake, Allwclean
  kodesChemistryModel/
    Allwmake, Allwclean         the model, then the KODES libraries
    kodesChemistryModel.{C,H}   the model
    kodesSpecieMap.{C,H}        KODES/pyJac <-> OpenFOAM specie ordering
    makeKodesChemistry*Types.*  runtime selection for psi/rhoReactionThermo
    pyJacSource/<m>/out/        pyJac's CUDA output, one directory per mechanism
tutorials/
  mechanisms/<m>/               the three mechanisms, shared by every case
  <case>/0.<m>/                 one case each, initial conditions per mechanism
utilities/
  kodesCase.py                  set a case up for a mechanism and variant, run it (foamlib)
  parametricStudy.ipynb         mechanisms x standard/kodes x balancers, on one case
  ...                           mesh generation and mechanism conversion/checking
requirements.txt                the Python side, for .venv
external/
  KODES/                        the KODES submodule, 2.0.1 (branch kodes-v2)
  pyJac/, Kinetic-Mechanisms/   only to regenerate a mechanism
```

## Building

Needs OpenFOAM v2412 or later sourced; for the KODES libraries also CMake ≥ 3.18 and
nvcc. Without nvcc the model is still built and the KODES libraries are skipped with a
note saying so.

```bash
git submodule update --init external/KODES

./Allwmake                              # the model + libkodes_<m>.so for every pyJacSource/<m>
KODES_MECHANISMS=grimech ./Allwmake     # the model + libkodes_grimech.so only
./Allwclean
```

The usual `Allwmake` options apply (`-j`, `-k`, `-log`), and `-prefix=` decides where
the libraries go, the model and the KODES libraries alike:

| Command | Install location |
|---|---|
| `./Allwmake` | `$FOAM_USER_LIBBIN` |
| `./Allwmake -prefix=group` | `$FOAM_SITE_LIBBIN` |
| `./Allwmake -prefix=openfoam` | `$FOAM_LIBBIN` |
| `./Allwmake -prefix=/some/pathname` | `/some/pathname/lib` |

The KODES libraries always go next to the model, so a case that finds the one finds the
other. Two variables steer them: `NVARCH` is the compute capability they are built for
(default `120`, an RTX 50xx; `NVARCH=80` for an A100), and `CUDACXX` the nvcc to use if
it is not on the `PATH` or in `/usr/local/cuda`.

### As an OpenFOAM plugin

```bash
cd $WM_PROJECT_DIR/plugins
git clone --recursive <this repository> kodes4foam
./Allwmake                              # builds every plugin, into $FOAM_LIBBIN
```

Built from `plugins/Allwmake` it goes to the project locations, like every plugin;
built on its own, to the user ones.

### The Python side

The tutorials are set up and run with [foamlib](https://github.com/gerlero/foamlib), from
a virtual environment in the repository:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

tutorials/2D_planar_flame/Allrun grimech kodes-temperature   # mechanism, chemistry variant

.venv/bin/jupyter lab utilities/parametricStudy.ipynb
```

[tutorials/README.md](tutorials/README.md) has the cases, mechanisms and variants.

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
    gatherCells       true;         // parallel runs: one ensemble on the master
}
```

The full list of `kodesSetting` entries is in the header of
[kodesChemistryModel.H](src/kodesChemistryModel/kodesChemistryModel.H). The species of
the library are matched by name against the case's species table when the case starts;
a library built for another mechanism stops the run there, naming the specie it lacks.

## Parallel runs

With `gatherCells true` (the default) every rank packs its cells, the master gathers
them with one `MPI_Gatherv`, integrates all of them as one KODES ensemble and scatters
the result back. Only the master opens the GPU.

That is what one GPU shared by several ranks wants: one context instead of one per
rank taking turns on the device, the device's memory planned once instead of each rank
taking a share of what the previous ones left, the whole mesh balanced together, and
no rank waiting on another's chemistry. It also makes the chemistry independent of the
decomposition, since which cells share a block no longer depends on which rank holds
them. The price is the state of every cell travelling to the master and back on every
chemistry step, about `(species + 1) x 8` bytes per cell each way.

`gatherCells false` integrates each rank's cells on its own, each rank opening the GPU
it sees - right when every rank has a GPU of its own (`CUDA_VISIBLE_DEVICES` per rank).

## Adding a mechanism

Generate pyJac's CUDA output into `src/kodesChemistryModel/pyJacSource/<name>/out`
(`python -m pyjac -l cuda ... -b <that directory>`) and run
`KODES_MECHANISMS=<name> ./Allwmake`. Cases then select it with `mechanism <name>;`.

For the tutorials, put its OpenFOAM `reactions` and `thermo.compressibleGas` in
`tutorials/mechanisms/<name>/` (`utilities/chemkinToFoam.py` writes both), and give a case
a `0.<name>` and a `constant/<name> -> ../../mechanisms/<name>` symlink; `./Allrun <name>`
and the parametric study pick it up from there.

Mind the names: KODES's own `data/mechanisms/h2o2` (9 species) and `pyJacSource/h2o2`
here (10, with N2) both build `libkodes_h2o2.so`. Installing KODES's default build next to
the model would replace this one.

## License

GNU General Public License v3, as OpenFOAM.
