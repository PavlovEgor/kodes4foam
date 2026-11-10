# 2D_flat_burner

The flat porous burner with a nitrogen co-flow of the laboratory of dynamics of reacting
systems of the Lebedev Physical Institute (<https://td.lpi.ru/labs1/DRS/>), in the regime
of relaxational diffusive-thermal oscillations of a rich methane-air flame, set up after
the computation in [Li et al., Combust. Flame 276 (2025) 114115](../../doc/) - every
number below is from its Sections 2 and 3 unless it says otherwise.

It runs with `reactingCanteraFoam`, as that computation needs what `reactingFoam` lacks:
a diffusion coefficient for each specie, thermal diffusion, radiation, buoyancy, and the
flux boundary condition on the surface of the burner.

## The burner

```
   x ^            outlet, 1 bar
     |      +------------------------+
     |      |                        |
     |      |                        |  outerZone, open to air at 1 bar and 293 K
   L |      |                        |
     |      |                        |
     |      |   flame                |
     |      +-------+-+-----+-+------+
     0      mainInlet w coflow w  outerZone
            0      RI       RO      RMAX   -> y, the radius
```

| | |
|---|---|
| `mainInlet`, 0 to RI = 7.48 mm | the porous plug, 0.2 mm channels 0.4 mm apart: CH4/air, phi = 1.15, M = 0.1 kg/m2/s, 293 K |
| `burnerWall`, w = 1.61 mm | the rims on both sides of the co-flow ring, no slip, 293 K |
| `coflowInlet`, to RO = 12.06 mm | the porous ring: N2, mN2 = 0.094 kg/m2/s, 293 K |
| `outerZone`, to RMAX = 22.44 mm, and the side | open: total pressure 1 bar, air comes in at 293 K |
| `outlet`, L = 224.4 mm | 1 bar |

The mass fractions of the main jet are CH4 0.0629, O2 0.2183, N2 0.7188, those of air
O2 0.232, N2 0.768. The paper has mN2 = 0, 0.037, 0.094 and 0.22 kg/m2/s; `massFlux` of
`coflowInlet` in `0.grimech/U` is the one to change. At 0.22 the secondary diffusion
flame is gone and the flame is closest to one-dimensional.

The plug and the ring are inlets, not resolved channels, as in the paper. The burner is
kept at 20 C by a thermostat, so its surface and the gases leaving it are at 293 K.

## Models

| | Here | The paper (EBIdnsFoam) |
|---|---|---|
| Mechanism | GRI-Mech 3.0 | the same |
| Diffusion | mixture-averaged, from Cantera, correction flux | the same |
| Thermal diffusion | Cantera's multicomponent coefficients | "Soret diffusion is considered", model not given |
| Radiation | optically thin, H2O, CO2, CO, CH4, TNF Planck mean coefficients | optically thin, the same four species |
| Buoyancy | yes, g = 9.81 m/s2 against x | yes |
| Species on `mainInlet` | `burnerSpecieFlux`: convection + diffusion = M Y0 | the same (its Eq. 2) |
| Velocity on `mainInlet` | `massFluxInletVelocity`: u = M/rho | the same (its Eq. 3) |
| Cells in the flame zone | 0.05 mm, 86 688 cells | 0.025 mm |
| Schemes | second order, `backward` | fourth order in space, second in time |
| Time step | Co < 0.3, at most 1 us | Co < 0.2 |

The mesh is the one difference that matters: the thermal thickness of this flame is
0.56 mm, so 0.05 mm puts 11 cells across it where the paper has 22. The paper's
resolution, 267 930 cells, is three times as many cells and half the time step; a coarse
mesh for tests, 0.1 mm and 30 275 cells:

```bash
../../utilities/flatBurnerBlockMesh.py --dx 0.025 system/blockMeshDict    # the paper's
../../utilities/flatBurnerBlockMesh.py --dx 0.1 system/blockMeshDict      # for tests
```

The time step is held to 1 us. With 10 us the pressure waves are not resolved and on
a coarse mesh T fell 10 K below that of the inflow on the axis, 100 to 200 mm above the
burner, by 1.5 ms. `limitTemperature` in `system/fvOptions` keeps T above 285 K, so the
log is not flooded with warnings from OpenFOAM below the 280 K `Tlow`. Its
`LimitedCells` in the log should stay at 0.

`burnerSpecieFlux` is on `coflowInlet` as well, which the paper does not say; with a
fixed value there the species that diffuse to the ring would leave through it.

Where the geometry of the paper leaves a choice: RO is taken as the outer radius of the
co-flow ring, which makes it 2.97 mm wide (the experimental section says 4 mm), and the
burner plane beyond the outer rim is open, as its Fig. 2 draws it.

## Running

```bash
./Allrun                    # grimech, the chemistry model OpenFOAM ships
./Allrun grimech kodes      # KODES on the GPU
./Allrun grimech standard --np 4
```

The case starts from the flame itself. `setFieldsDict` lays the one-dimensional
burner-stabilised flame Cantera finds for the plug (M, the mixture and 293 K of the plug,
the same transport, Soret and radiation) over 0 < r < RI: T, 16 species and u = M/rho,
one layer per point of its grid, the products of its last point up to the outlet. It
burns at 0.83 mm and peaks at 1774 K. Density and velocity are in step from the first
step, so no pressure wave starts, and there is no ignition to wait for. The dictionary
and the species files in `0.grimech` it needs are written by

```bash
../../utilities/burnerFlameSetFields.py .      # Cantera, ~20 s
```

which is to be run again when M, the mixture or the mechanism change. `endTime`
is 50 ms, three to four periods of the oscillation the paper finds at 69 Hz;
`postProcessing/flame/0/volFieldValue.dat` has max(T) and max(OH) of every step, which is
where the period shows. With the standard chemistry model that is days on one core: the
case is meant for `kodes`.
