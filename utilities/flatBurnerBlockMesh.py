#!/usr/bin/env python3
"""
Generate the blockMeshDict of tutorials/2D_flat_burner: the flat porous burner
with a nitrogen co-flow of the Lebedev Physical Institute, as it was computed
in Li et al., Combust. Flame 276 (2025) 114115, Fig. 2.

The domain is an axisymmetric wedge, x along the axis and y the radius. In the
burner plane x = 0, from the axis outwards:

    mainInlet       0  .. RI            the porous plug, methane-air
    burnerWall      RI .. RI + w        the rim between the two streams
    coflowInlet     RI + w .. RO        the porous ring, nitrogen
    burnerWall      RO .. RO + w        the outer rim
    outerZone       RO + w .. RMAX      open to the surrounding air

and the side y = RMAX is outerZone too, the top x = L the outlet.

The cells are dx by dx up to the height of the refined zone and out to the
outer rim. Above the zone they grow by --growth per cell, beyond the rim to
--outer-ratio times dx at RMAX. The paper has dx = 0.025 mm; the default here
is four times that, about 30 000 cells instead of half a million.

Usage:
    flatBurnerBlockMesh.py [options] <output blockMeshDict>
"""

import argparse
import math
from pathlib import Path

HEADER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2412                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def graded(length, first, ratio):
    """
    (cells, last/first) of a run of cells over length that starts at first
    and grows to about ratio times that.
    """
    for n in range(2, 100000):
        r = ratio ** (1 / (n - 1))
        if first * (r**n - 1) / (r - 1) >= length:
            return n, ratio
    raise ValueError("no grading found")


def growing(length, first, growth):
    """(cells, last/first) of a run of cells growing by growth per cell."""
    n = math.ceil(math.log(1 + length * (growth - 1) / first) / math.log(growth))
    n = max(n, 2)
    return n, growth ** (n - 1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output")
    parser.add_argument("--dx", type=float, default=0.1, help="cell size in the refined zone [mm] (default 0.1; the paper: 0.025)")
    parser.add_argument("--RI", type=float, default=7.48, help="radius of the main inlet [mm]")
    parser.add_argument("--RO", type=float, default=12.06, help="outer radius of the co-flow inlet [mm]")
    parser.add_argument("--wall", type=float, default=1.61, help="thickness of the two rims [mm]")
    parser.add_argument("--RMAX", type=float, default=22.44, help="radius of the domain [mm]")
    parser.add_argument("--L", type=float, default=224.4, help="height of the domain [mm]")
    parser.add_argument("--zone", type=float, default=6.0, help="height of the refined zone [mm]")
    parser.add_argument("--growth", type=float, default=1.04, help="axial growth per cell above the zone")
    parser.add_argument("--outer-ratio", type=float, default=5.0, help="radial cell size at RMAX over dx")
    parser.add_argument("--angle", type=float, default=5.0, help="wedge angle [deg]")
    a = parser.parse_args()

    radii = [0.0, a.RI, a.RI + a.wall, a.RO, a.RO + a.wall, a.RMAX]
    heights = [0.0, a.zone, a.L]
    bottom = ["mainInlet", "burnerWall", "coflowInlet", "burnerWall", "outerZone"]

    if sorted(radii) != radii or len(set(radii)) != len(radii):
        raise SystemExit("the radii have to increase: " + str(radii))

    # Cells and grading across each radial and axial stretch
    nr = [(max(round((r1 - r0) / a.dx), 1), 1.0) for r0, r1 in zip(radii[:-2], radii[1:-1])]
    nr.append(graded(radii[-1] - radii[-2], a.dx, a.outer_ratio))
    nx = [
        (max(round(a.zone / a.dx), 1), 1.0),
        growing(a.L - a.zone, a.dx, a.growth),
    ]

    half = math.radians(a.angle / 2)
    c, s = math.cos(half), math.sin(half)

    vertices = []
    index = {}  # (axial, radial, side) -> vertex; side 0 back, 1 front

    for i, x in enumerate(heights):
        for j, r in enumerate(radii):
            for side in (0, 1):
                if j == 0 and side == 1:
                    index[i, j, 1] = index[i, j, 0]
                    continue
                index[i, j, side] = len(vertices)
                z = r * s * (1 if side else -1) if j else 0.0
                vertices.append((x, r * c, z))

    def v(i, j, side):
        return index[i, j, side]

    blocks = []
    for i in range(len(heights) - 1):
        for j in range(len(radii) - 1):
            corners = [
                v(i, j, 0), v(i + 1, j, 0), v(i + 1, j + 1, 0), v(i, j + 1, 0),
                v(i, j, 1), v(i + 1, j, 1), v(i + 1, j + 1, 1), v(i, j + 1, 1),
            ]
            blocks.append(
                "    hex ({}) ({} {} 1) simpleGrading ({:.6g} {:.6g} 1)".format(
                    " ".join(map(str, corners)), nx[i][0], nr[j][0], nx[i][1], nr[j][1]
                )
            )

    patches = {}

    def face(patch, kind, *corners):
        patches.setdefault(patch, (kind, []))[1].append(
            "(" + " ".join(map(str, corners)) + ")"
        )

    top = len(heights) - 1
    for j in range(len(radii) - 1):
        kind = "wall" if bottom[j] == "burnerWall" else "patch"
        face(bottom[j], kind, v(0, j, 0), v(0, j, 1), v(0, j + 1, 1), v(0, j + 1, 0))
        face("outlet", "patch", v(top, j, 0), v(top, j + 1, 0), v(top, j + 1, 1), v(top, j, 1))

    out = len(radii) - 1
    for i in range(len(heights) - 1):
        face("outerZone", "patch", v(i, out, 0), v(i, out, 1), v(i + 1, out, 1), v(i + 1, out, 0))
        face("axis", "empty", v(i, 0, 0), v(i + 1, 0, 0), v(i + 1, 0, 0), v(i, 0, 0))
        for j in range(len(radii) - 1):
            face("back", "wedge", v(i, j, 0), v(i, j + 1, 0), v(i + 1, j + 1, 0), v(i + 1, j, 0))
            face("front", "wedge", v(i, j, 1), v(i + 1, j, 1), v(i + 1, j + 1, 1), v(i, j + 1, 1))

    cells = sum(n for n, _ in nx) * sum(n for n, _ in nr)

    lines = [HEADER]
    lines.append(f"// utilities/flatBurnerBlockMesh.py --dx {a.dx:g}: {cells} cells,")
    lines.append(f"// {sum(n for n, _ in nx)} along the axis by {sum(n for n, _ in nr)} across")
    lines.append("")
    lines.append("scale   0.001;")
    lines.append("")
    lines.append("vertices")
    lines.append("(")
    for k, (x, y, z) in enumerate(vertices):
        lines.append(f"    ({x:.10g} {y:.10g} {z:.10g})  // {k}")
    lines.append(");")
    lines.append("")
    lines.append("blocks")
    lines.append("(")
    lines.extend(blocks)
    lines.append(");")
    lines.append("")
    lines.append("edges")
    lines.append("(")
    lines.append(");")
    lines.append("")
    lines.append("boundary")
    lines.append("(")
    for name, (kind, faces) in patches.items():
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type {kind};")
        lines.append("        faces")
        lines.append("        (")
        lines.extend(f"            {f}" for f in faces)
        lines.append("        );")
        lines.append("    }")
    lines.append(");")
    lines.append("")
    lines.append("mergePatchPairs")
    lines.append("(")
    lines.append(");")
    lines.append("")
    lines.append("// " + "*" * 73 + " //")

    Path(a.output).write_text("\n".join(lines) + "\n")
    print(f"{a.output}: {cells} cells")


if __name__ == "__main__":
    main()
