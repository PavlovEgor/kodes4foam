#!/usr/bin/env python3
"""
Write the setFieldsDict that starts tutorials/2D_flat_burner from the flame
itself: the one-dimensional burner-stabilised flame Cantera finds for the plug
(BurnerFlame: the mass flux, the composition and the temperature of the plug,
mixture-averaged diffusion, Soret, optically thin radiation) is laid over the
plug, 0 < r < RI, as layers of boxToCell, one per point of Cantera's grid.

Each layer has T, the mass fractions and the axial velocity u = M/rho of that
point, so the flame starts where it burns, with the density and the velocity
in step and nothing to push a pressure wave out. Above the last point, up to
the outlet, the layer is the products as Cantera leaves them. Outside the plug
the case stays as setFields defaults it: air at 293 K.

Species: those of the case's 0.<mechanism> and every other one whose mass
fraction somewhere in the flame is above --min-Y; a file for each new one is
made in 0.<mechanism> from its Ydefault. What the species left out carry is
given to the inert specie.

    ../../utilities/burnerFlameSetFields.py .          # from the case
"""

import argparse
import re
from pathlib import Path

import cantera as ct

FIELDS_OF_STATE = ("T", "U")


def solve_flame(mechanism, T, p, Y, mdot, width, radiation, soret):
    gas = ct.Solution(str(mechanism))
    gas.TPY = T, p, Y

    flame = ct.BurnerFlame(gas, width=width)
    flame.burner.mdot = mdot
    flame.transport_model = "mixture-averaged"
    flame.radiation_enabled = radiation
    flame.set_refine_criteria(ratio=3, slope=0.05, curve=0.1, prune=0.01)
    flame.solve(loglevel=0, auto=True)

    if soret:
        # Thermal diffusion on a solution that has converged without it
        flame.soret_enabled = True
        flame.solve(loglevel=0)

    return flame


def boxes(x, top):
    """The edges of the layers: midway between the points, the last to top."""
    edges = [-1.0]
    edges += [0.5 * (a + b) for a, b in zip(x[:-1], x[1:])]
    edges.append(top)
    return list(zip(edges[:-1], edges[1:]))


def fmt(v):
    return f"{v:.6g}"


def header(obj):
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2412                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def species_files(zero):
    return {
        p.name for p in zero.iterdir()
        if p.is_file() and p.name not in ("p", "p_rgh", "Ydefault") + FIELDS_OF_STATE
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("case", type=Path)
    ap.add_argument("--mechanism", default="grimech")
    ap.add_argument("--mdot", type=float, default=0.1, help="M [kg/m2/s]")
    ap.add_argument("--T", type=float, default=293.0, help="the plug [K]")
    ap.add_argument("--p", type=float, default=1e5, help="[Pa]")
    ap.add_argument("--Y", default="CH4:0.0629, O2:0.2183, N2:0.7188")
    ap.add_argument("--inert", default="N2")
    ap.add_argument("--RI", type=float, default=7.48e-3, help="plug radius [m]")
    ap.add_argument("--top", type=float, default=1.0,
                    help="the last layer reaches this x [m], past the outlet")
    ap.add_argument("--width", type=float, default=0.02,
                    help="Cantera's domain [m]")
    ap.add_argument("--min-Y", type=float, default=1e-4)
    ap.add_argument("--no-radiation", action="store_true")
    ap.add_argument("--no-soret", action="store_true")
    args = ap.parse_args()

    case = args.case.resolve()
    zero = case / f"0.{args.mechanism}"
    flame = solve_flame(
        case / "constant" / args.mechanism / f"{args.mechanism}.yaml",
        args.T, args.p, args.Y, args.mdot, args.width,
        not args.no_radiation, not args.no_soret,
    )

    names = flame.gas.species_names
    Y = flame.Y
    have = species_files(zero)
    chosen = sorted(
        (have | {s for i, s in enumerate(names) if Y[i].max() > args.min_Y})
        - {args.inert}
    )

    for s in chosen:
        if s not in have:
            text = (zero / "Ydefault").read_text()
            (zero / s).write_text(
                re.sub(r"(object\s+)Ydefault;", rf"\g<1>{s};", text)
            )

    x = flame.grid
    u = args.mdot / flame.density
    idx = {s: names.index(s) for s in chosen}

    out = [header("setFieldsDict")]
    out.append(
        f"// Written by utilities/burnerFlameSetFields.py: Cantera {ct.__version__}"
        f" BurnerFlame,\n// M = {args.mdot} kg/m2/s, {args.Y}, {args.T} K,"
        f" {len(x)} points, Tmax = {flame.T.max():.0f} K\n\n"
    )
    out.append("defaultFieldValues\n(\n    volScalarFieldValue T 293\n")
    air = {"O2": 0.232, args.inert: 0.768}
    for s in chosen + [args.inert]:
        out.append(f"    volScalarFieldValue {s} {fmt(air.get(s, 0))}\n")
    out.append(");\n\nregions\n(\n")

    for i, (lo, hi) in enumerate(boxes(x, args.top)):
        ys = {s: max(float(Y[idx[s], i]), 0.0) for s in chosen}
        ys[args.inert] = 1.0 - sum(ys.values())
        out.append(
            f"    boxToCell\n    {{\n        box ({fmt(lo)} -1 -1)"
            f" ({fmt(hi)} {fmt(args.RI)} 1);\n        fieldValues\n        (\n"
            f"            volScalarFieldValue T {fmt(flame.T[i])}\n"
            f"            volVectorFieldValue U ({fmt(u[i])} 0 0)\n"
        )
        for s in chosen + [args.inert]:
            out.append(f"            volScalarFieldValue {s} {fmt(ys[s])}\n")
        out.append("        );\n    }\n")

    out.append(");\n\n\n// ************************************************************************* //\n")
    (case / "system" / "setFieldsDict").write_text("".join(out))

    q = flame.heat_release_rate
    print(
        f"{len(x)} points, Tmax {flame.T.max():.0f} K, T at {x[-1]*1e3:.0f} mm"
        f" {flame.T[-1]:.0f} K, heat release peak at {x[q.argmax()]*1e3:.2f} mm,"
        f" u {u[0]:.4f} -> {u[-1]:.3f} m/s\n"
        f"species: {' '.join(chosen)} + {args.inert}"
    )


if __name__ == "__main__":
    main()
