# 1D_flat_burner

The centre line of [2D_flat_burner](../2D_flat_burner/README.md) on its own: the
methane-air flame on the porous plug, phi = 1.15, M = 0.1 kg/m2/s, 293 K, in one
dimension, as [Li et al., Combust. Flame 276 (2025) 114115](../../doc/) also computed
it. Its 1D runs oscillate in the relaxational regime at 65 Hz, without buoyancy: the
front comes to 0.68 mm from the plug, dies down and is blown to 2.73 mm, lights again
and runs back.

Everything but the geometry is the 2D case's: `reactingCanteraLMFoam`, GRI-Mech 3.0,
mixture-averaged diffusion with Soret, optically thin radiation, `burnerSpecieFlux` and
`massFluxInletVelocity` on `mainInlet`. g is zero, as in the paper's 1D runs.

| | |
|---|---|
| Domain | 0 to 15 mm, 600 cells of 0.025 mm, the paper's resolution |
| `mainInlet` | the plug, as in the 2D case |
| `outlet` | 1 bar |
| Start | the Cantera burner flame, `setFieldsDict` from `utilities/burnerFlameSetFields.py` |
| `endTime` | 100 ms, about six periods |

`postProcessing/flame/0/fieldMinMax.dat` has max(T), max(OH) and max(Qdot) of every
step with where they are; x of max(Qdot) is the flame front. OH of this rich flame
peaks in the products, not at the front.

```bash
./Allrun                        # standard chemistry; ~600 cells, a few CPU hours
./Allrun grimech standard --np 4
```
