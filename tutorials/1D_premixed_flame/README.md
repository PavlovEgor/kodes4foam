# 1D_premixed_flame

A freely propagating stoichiometric H2/air flame, 320 K, 1 atm, in a 12 mm duct of 600
cells, with `reactingCanteraFoam`. It is the check of the solver's molecular transport:
the laminar flame speed depends on it and on little else, and Cantera computes the same
flame with the same mechanism and the same transport coefficients.

The mixture enters at 2.5 m/s, close to the flame speed, so the front that the products
downstream of x = 4 mm start stays in the duct. The outlet is `waveTransmissive`: with
the pressure fixed there the duct is an acoustic resonator the flame drives, and the
front moves with a pressure wave of several kPa instead of at the flame speed.

```bash
./Allrun                                        # h2o2; 4 minutes on one core
../../utilities/flameSpeed.py .                 # flame speed, and Cantera's
```

```
1D_premixed_flame: T = 320 K, p = 101325 Pa, Y = {'H2': 0.028522, 'N2': 0.745124, 'O2': 0.226354}
  front at 4.114 mm, moving at -0.0113 m/s (fit over 0.001 .. 0.003 s, 41 times)
  flame speed        2.5113 m/s
  Cantera            2.3196 m/s  (multicomponent, Soret on)
  difference         +8.3 %
```

`flameSpeed.py` takes the flame speed as the inlet velocity less the velocity of the
front, and runs Cantera's `FreeFlame` for what `constant/canteraProperties` and `0/` say.

## What it gives

`diffusion` and `Soret` in `constant/canteraProperties`, flame speed in m/s:

| | Cantera | 600 cells, 20 um | 1200 cells, 10 um |
|---|---|---|---|
| `mixtureAveraged`, `Soret off` | 2.573 | 2.636 (+2.4 %) | 2.596 (+0.9 %) |
| `multicomponent`, `Soret on` | 2.320 | 2.440 (+5.2 %) | 2.374 (+2.3 %) |
| `mixtureAveraged`, `Soret on` | - | 2.511 | |

The difference from Cantera halves with the cell size, in both rows: it is the mesh and
the first-order time stepping, not the transport model. Thermal diffusion slows this
flame by a tenth, which is why it is on in `2D_flat_burner`.

Cantera has thermal diffusion with multicomponent transport only, so the case as it is
checked in, mixture-averaged with Soret, has no reference of its own; the 8 % above are
against the multicomponent flame, of which about 3 % are the diffusion model.

`0.grimech` is the same flame with GRI-Mech 3.0, 53 species instead of 10.
