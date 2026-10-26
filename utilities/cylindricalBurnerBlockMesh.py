#!/usr/bin/env python3
"""
Generate a blockMeshDict for an axisymmetric-topology burner: a stack of
cylindrical segments, each divided into concentric rings, meshed as a
butterfly (O-grid) so there is no singular cell on the axis.

A segment may carry fewer rings than the one above it, which is how a sudden
expansion is built: the rings that only exist above the step get their bottom
faces turned into a wall (the backplane), and the outer face of the narrower
segment becomes the tube wall.  That is the whole topology needed for a swirl
burner discharging into a confinement chamber, and for a coaxial injector
whose streams enter through concentric patches of one inlet plane.

The mesh is all hexahedra.  Each ring contributes four blocks (one per
quadrant) and the core contributes one, so a stack of S segments over N rings
is S*(4N+1) blocks.

Geometry is described by a small JSON file:

    {
      "scale": 1,
      "rings":    [0.017, 0.05],           # outer radius of each ring [m]
      "coreFraction": 0.5,                 # core square corner / rings[0]
      "nCore": 16, "nAzimuthal": 16,       # cells across the core / per quadrant
      "nRadial": [8, 12],                  # cells across each ring
      "segments": [
        {"z0": -0.04, "z1": 0.0,  "nz": 40,  "rings": 1,
         "inlet": {"0": "inlet"}},         # ring index -> bottom patch name
        {"z0":  0.0,  "z1": 0.15, "nz": 100, "rings": 2}
      ],
      "outletPatch": "outlet",
      "wallPatch": "wall",
      "stepPatch": "backplane"
    }

"rings" inside a segment is how many of the global rings exist there; ring 0
is always the O-grid core.  A segment's "inlet" maps a ring index (as a
string, since JSON keys are strings) to the patch its bottom faces go to, and
is only meaningful for the bottom-most segment; unlisted rings there fall to
the step patch.

Usage:
    cylindricalBurnerBlockMesh.py <geometry.json> <output blockMeshDict>
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Quadrant corner angles.  Points sit on the diagonals so that the four arc
# edges of a circle have their midpoints on the axes, where they are easiest
# to read and to check.
CORNER_ANGLES = [45.0, 135.0, 225.0, 315.0]


class Geometry:
    def __init__(self, spec):
        self.scale = spec.get("scale", 1)
        self.rings = spec["rings"]
        self.core_fraction = spec.get("coreFraction", 0.5)
        self.n_core = spec["nCore"]
        self.n_azimuthal = spec["nAzimuthal"]
        self.n_radial = spec["nRadial"]
        self.segments = spec["segments"]
        self.outlet_patch = spec.get("outletPatch", "outlet")
        self.wall_patch = spec.get("wallPatch", "wall")
        self.step_patch = spec.get("stepPatch", "backplane")

        if len(self.n_radial) != len(self.rings):
            sys.exit(
                f"error: nRadial has {len(self.n_radial)} entries but there "
                f"are {len(self.rings)} rings"
            )
        for r0, r1 in zip(self.rings, self.rings[1:]):
            if r1 <= r0:
                sys.exit(f"error: ring radii must increase: {self.rings}")

        # z levels: one more than there are segments, and they have to join up
        self.z_levels = [self.segments[0]["z0"]]
        for s in self.segments:
            if abs(s["z0"] - self.z_levels[-1]) > 1e-12:
                sys.exit(
                    f"error: segment starting at z = {s['z0']} does not meet "
                    f"the previous one, which ended at {self.z_levels[-1]}"
                )
            self.z_levels.append(s["z1"])

    @property
    def n_rings(self):
        return len(self.rings)

    def radius(self, k):
        """Radius of radial level k; level 0 is the core square's corners."""
        return self.core_fraction * self.rings[0] if k == 0 else self.rings[k - 1]

    def vertex_id(self, j, k, q):
        """Index of the vertex at z level j, radial level k, quadrant q.

        Every (j, k, q) is emitted whether or not a block uses it, so this is
        plain arithmetic rather than a lookup that has to track which rings
        exist where. blockMesh does not mind unused vertices."""
        return (j * (self.n_rings + 1) + k) * 4 + (q % 4)


def point(radius, angle_deg, z):
    a = math.radians(angle_deg)
    return (radius * math.cos(a), radius * math.sin(a), z)


def fmt(p):
    return "( %.10g %.10g %.10g )" % p


def build_vertices(g):
    lines = []
    for j, z in enumerate(g.z_levels):
        for k in range(g.n_rings + 1):
            for q, angle in enumerate(CORNER_ANGLES):
                p = point(g.radius(k), angle, z)
                lines.append(
                    "    %-28s // %3d  z%d r%d q%d"
                    % (fmt(p), g.vertex_id(j, k, q), j, k, q)
                )
    return lines


def build_blocks(g):
    lines = []
    for s, seg in enumerate(g.segments):
        j0, j1, nz = s, s + 1, seg["nz"]
        grading = seg.get("grading", 1)

        lines.append(
            "    // segment %d: z %g -> %g, %d ring(s)"
            % (s, seg["z0"], seg["z1"], seg["rings"])
        )

        core = [g.vertex_id(j0, 0, q) for q in range(4)] + \
               [g.vertex_id(j1, 0, q) for q in range(4)]
        lines.append(
            "    hex (%s) (%d %d %d) simpleGrading (1 1 %s)"
            % (" ".join(map(str, core)), g.n_core, g.n_core, nz, grading)
        )

        for k in range(1, seg["rings"] + 1):
            for q in range(4):
                v = [
                    g.vertex_id(j0, k - 1, q), g.vertex_id(j0, k, q),
                    g.vertex_id(j0, k, q + 1), g.vertex_id(j0, k - 1, q + 1),
                    g.vertex_id(j1, k - 1, q), g.vertex_id(j1, k, q),
                    g.vertex_id(j1, k, q + 1), g.vertex_id(j1, k - 1, q + 1),
                ]
                radial_grading = seg.get("radialGrading", {}).get(str(k), 1)
                lines.append(
                    "    hex (%s) (%d %d %d) simpleGrading (%s 1 %s)"
                    % (" ".join(map(str, v)), g.n_radial[k - 1],
                       g.n_azimuthal, nz, radial_grading, grading)
                )
    return lines


def build_edges(g):
    """One arc per quadrant per circle, so the ring boundaries are round
    rather than the inscribed squares the block edges would otherwise give.

    Only for circles a block actually reaches: an arc on a ring that exists in
    neither the segment below a z level nor the one above it corresponds to no
    block edge, and blockMesh rejects the topology over it."""
    lines = []
    for j, z in enumerate(g.z_levels):
        rings_here = max(
            (g.segments[s]["rings"] for s in (j - 1, j) if 0 <= s < len(g.segments)),
            default=0,
        )
        for k in range(1, rings_here + 1):
            r = g.radius(k)
            for q in range(4):
                mid = point(r, CORNER_ANGLES[q] + 45.0, z)
                lines.append(
                    "    arc %d %d %s"
                    % (g.vertex_id(j, k, q), g.vertex_id(j, k, q + 1), fmt(mid))
                )
    return lines


def bottom_faces(g, j, k):
    """Faces of radial level k at z level j, normal pointing in -z."""
    if k == 0:
        return ["(%d %d %d %d)" % tuple(g.vertex_id(j, 0, q) for q in (0, 3, 2, 1))]
    return [
        "(%d %d %d %d)" % (
            g.vertex_id(j, k - 1, q), g.vertex_id(j, k - 1, q + 1),
            g.vertex_id(j, k, q + 1), g.vertex_id(j, k, q),
        )
        for q in range(4)
    ]


def top_faces(g, j, k):
    """Faces of radial level k at z level j, normal pointing in +z."""
    if k == 0:
        return ["(%d %d %d %d)" % tuple(g.vertex_id(j, 0, q) for q in (0, 1, 2, 3))]
    return [
        "(%d %d %d %d)" % (
            g.vertex_id(j, k - 1, q), g.vertex_id(j, k, q),
            g.vertex_id(j, k, q + 1), g.vertex_id(j, k - 1, q + 1),
        )
        for q in range(4)
    ]


def outer_faces(g, s, k):
    """Faces on the outside of ring k over segment s, normal pointing out."""
    j0, j1 = s, s + 1
    return [
        "(%d %d %d %d)" % (
            g.vertex_id(j0, k, q), g.vertex_id(j0, k, q + 1),
            g.vertex_id(j1, k, q + 1), g.vertex_id(j1, k, q),
        )
        for q in range(4)
    ]


def build_boundary(g):
    patches = {}

    def add(name, kind, faces):
        entry = patches.setdefault(name, {"type": kind, "faces": []})
        entry["faces"].extend(faces)

    first, last = g.segments[0], g.segments[-1]

    # Inlet plane: the bottom of the lowest segment, split by ring
    inlet_map = {int(k): v for k, v in first.get("inlet", {}).items()}
    for k in range(first["rings"] + 1):
        name = inlet_map.get(k)
        if name:
            add(name, "patch", bottom_faces(g, 0, k))
        else:
            add(g.step_patch, "wall", bottom_faces(g, 0, k))

    # Outlet: the top of the highest segment
    for k in range(last["rings"] + 1):
        add(g.outlet_patch, "patch", top_faces(g, len(g.segments), k))

    # Steps between segments, in either direction, and the outer wall of
    # whichever segment is the narrower one there
    for s in range(len(g.segments) - 1):
        below, above = g.segments[s]["rings"], g.segments[s + 1]["rings"]
        for k in range(min(below, above) + 1, max(below, above) + 1):
            # A ring present on one side only: its face at the joining z level
            # is a wall, facing into whichever segment has it
            if above > below:
                add(g.step_patch, "wall", bottom_faces(g, s + 1, k))
            else:
                add(g.step_patch, "wall", top_faces(g, s + 1, k))

    # Outer wall of every segment
    for s, seg in enumerate(g.segments):
        add(g.wall_patch, "wall", outer_faces(g, s, seg["rings"]))

    lines = []
    for name, entry in patches.items():
        lines.append("    %s" % name)
        lines.append("    {")
        lines.append("        type %s;" % entry["type"])
        lines.append("        faces")
        lines.append("        (")
        lines.extend("            %s" % f for f in entry["faces"])
        lines.append("        );")
        lines.append("    }")
    return lines


HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Generated by utilities/cylindricalBurnerBlockMesh.py from
//     {source}
// Do not edit by hand -- edit the geometry file and regenerate.
//
// {cells} cells: butterfly (O-grid) topology, all hexahedra, no cell on the
// axis with a degenerate face.

scale   {scale};
"""


def cell_count(g):
    total = 0
    for seg in g.segments:
        per_layer = g.n_core * g.n_core
        for k in range(1, seg["rings"] + 1):
            per_layer += 4 * g.n_radial[k - 1] * g.n_azimuthal
        total += per_layer * seg["nz"]
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("geometry", help="JSON geometry description")
    parser.add_argument("output", help="blockMeshDict to write")
    parser.add_argument(
        "--refine", type=float, default=1.0,
        help="Multiply every cell count by this and round up; 2 gives roughly "
             "eight times the cells (default: 1)",
    )
    args = parser.parse_args()

    spec = json.loads(Path(args.geometry).read_text())

    if args.refine != 1.0:
        r = args.refine
        spec["nCore"] = max(1, math.ceil(spec["nCore"] * r))
        spec["nAzimuthal"] = max(1, math.ceil(spec["nAzimuthal"] * r))
        spec["nRadial"] = [max(1, math.ceil(n * r)) for n in spec["nRadial"]]
        for seg in spec["segments"]:
            seg["nz"] = max(1, math.ceil(seg["nz"] * r))

    g = Geometry(spec)

    out = [HEADER.format(source=args.geometry, cells=f"{cell_count(g):,}", scale=g.scale)]
    out.append("\nvertices\n(")
    out.extend(build_vertices(g))
    out.append(");\n")
    out.append("blocks\n(")
    out.extend(build_blocks(g))
    out.append(");\n")
    out.append("edges\n(")
    out.extend(build_edges(g))
    out.append(");\n")
    out.append("boundary\n(")
    out.extend(build_boundary(g))
    out.append(");\n")
    out.append("mergePatchPairs\n(\n);\n")
    out.append("// ************************************************************************* //")

    Path(args.output).write_text("\n".join(out) + "\n")
    print(f"Wrote {args.output}")
    print(f"    {cell_count(g):,} cells, "
          f"{sum(4 * s['rings'] + 1 for s in g.segments)} blocks, "
          f"{len(g.z_levels) * (g.n_rings + 1) * 4} vertices")


if __name__ == "__main__":
    main()
