# uniformHeating2D

Homogeneous (0-D-like) autoignition case built from `counterFlowFlame2D`: same
GRI-Mech 3.0 mechanism, same thermophysical model and the same solver settings,
but on a square, fully periodic domain with zero velocity.

Because every cell starts with identical composition, temperature and pressure,
and the boundaries are cyclic, there are no gradients anywhere. Convection and
diffusion contribute nothing and each cell integrates exactly the same chemistry
— a constant-volume homogeneous reactor replicated over the whole mesh. This is
what makes the case useful as a chemistry-solver benchmark (`standard` vs
`kodesChemistryModel`): the physics is trivial, all the cost is chemistry.

## Setup

| | |
|---|---|
| Solver | `reactingFoam` |
| Domain | 20 × 20 × 1 mm square, 40 × 40 × 1 cells (1600 cells) |
| Boundaries | `cyclic` in x and y, `empty` front/back |
| Mechanism | GRI-Mech 3.0 (`constant/reactionsGRI`, 53 species) |
| Chemistry | `ode` / `seulex`, absTol 1e-12, relTol 1e-1 |
| Combustion | `laminar`, turbulence `laminar` |
| Initial state | T = 2000 K, p = 1 bar, U = 0 |
| Mixture | stoichiometric CH4/air — CH4 0.05437, O2 0.21749, N2 0.72814 (mass) |
| Time | deltaT = 1e-6 s, endTime = 1e-3 s |

The domain is closed, so this is combustion at constant volume: pressure rises
with the temperature instead of staying at 1 bar.

`adjustTimeStep` is left on for consistency with `counterFlowFlame2D`, but with
U = 0 the Courant number is zero, so the step is set entirely by `maxDeltaT`
(1e-6 s) and the run is effectively at fixed deltaT.

## Running

```sh
./Allrun      # blockMesh + reactingFoam
./Allclean
```

The `fieldMinMax` function object in `system/controlDict` prints min/max of `T`
and `p` every 10 steps — the quickest way to watch the ignition and to confirm
the field really stays uniform.

To switch to the GPU chemistry model, set `method kodesChemistryModel` in
`constant/chemistryProperties`; `kodesSetting` there names the KODES library.

## Expected behaviour

Ignition is essentially immediate at 2000 K: the temperature leaves 2000 K
within the first few microseconds and settles at the constant-volume adiabatic
flame temperature (~2900 K) well before the end of the run.

The min/max spread across cells is not exactly zero (order 0.1 K) — the cells
are decoupled physically, but the Yi/h/p equations are still solved iteratively
to a finite tolerance, so round-off differs slightly from cell to cell. A
relative spread of ~1e-5 is expected and is not a setup error.
