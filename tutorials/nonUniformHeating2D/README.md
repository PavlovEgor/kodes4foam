# nonUniformHeating2D

Homogeneous (0-D-like) autoignition: a stoichiometric hydrogen-air mixture at 1500 K in a
square, fully periodic box with zero velocity, built from `counterFlowFlame2D`'s
thermophysical model and solver settings.

Because every cell starts with identical composition, temperature and pressure, and the
boundaries are cyclic, there are no gradients anywhere. Convection and diffusion
contribute nothing and each cell integrates exactly the same chemistry — a
constant-volume homogeneous reactor replicated over the whole mesh. That makes it the
quickest check that a mechanism and a chemistry model run at all, and a benchmark where
all of the cost is chemistry. It is also the one case where batch balancing has nothing
to work with: every cell needs the same number of steps.

## Setup

| | |
|---|---|
| Solver | `reactingFoam` |
| Domain | 20 × 20 × 1 mm square, 128 × 128 × 1 cells (16 384 cells) |
| Boundaries | `cyclic` in x and y, `empty` front/back |
| Mechanisms | `h2o2`, `grimech`, `C1_C3_HT_NOX_159_2459` — `0.<mechanism>` for each |
| Chemistry | `ode` / `seulex`, absTol 1e-12, relTol 1e-2 |
| Combustion | `laminar`, turbulence `laminar` |
| Initial state | T = 1500 K, p = 1 atm, U = 0 |
| Mixture | stoichiometric H2/air — H2 0.02723, O2 0.21608, N2 0.75669 (mass) |
| Time | deltaT = 1e-6 s, endTime = 1e-3 s |

The domain is closed, so this is combustion at constant volume: pressure rises with the
temperature instead of staying at 1 atm.

`adjustTimeStep` is left on for consistency with `counterFlowFlame2D`, but with U = 0 the
Courant number is zero, so the step is set entirely by `maxDeltaT` (1e-6 s) and the run
is effectively at fixed deltaT.

## Running

```sh
./Allrun                            # as set up now: h2o2, standard
./Allrun grimech kodes-temperature  # another mechanism and chemistry variant
./Allclean
```

See [../README.md](../README.md) for the mechanisms and variants. The `fieldMinMax`
function object in `system/controlDict` prints min/max of `T` and `p` every 10 steps —
the quickest way to watch the ignition and to confirm the field really stays uniform.

## Expected behaviour

The min/max spread across cells is not exactly zero — the cells are decoupled
physically, but the Yi/h/p equations are still solved iteratively to a finite tolerance,
so round-off differs slightly from cell to cell. A relative spread of ~1e-5 is expected
and is not a setup error.
