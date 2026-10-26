# Reconstructed burner geometries

The two 3D tutorial cases ship no mesh: their meshes and fields are the external dataset
the top-level [README](../README.md) links to, and nothing could be run from what is in
this repository. The meshes used by the three mechanism sets are therefore
**reconstructions**, generated from the parameter files here by
[utilities/cylindricalBurnerBlockMesh.py](../../../utilities/cylindricalBurnerBlockMesh.py).

They are not the geometry the papers used. Read the `_comment` block at the top of each
JSON before comparing anything to a published number: it states which dimensions come
from the case README and which are assumed.

| File | Burner | Assumed from | Default cells |
|---|---|---|---|
| [AHEAD.json](AHEAD.json) | premixed swirl burner, sudden expansion into a confinement | 34 mm mixing chamber from the case README; chamber size and heights from its `sampleDict` | 207 616 |
| [HYLON.json](HYLON.json) | coaxial dual-swirl injector, non-premixed | nothing — the case README gives no dimensions, only a citation | 156 000 |

## Regenerating

```bash
./utilities/cylindricalBurnerBlockMesh.py \
    tutorials/DLBFoam-Hydrogen-Tutorials/geometry/AHEAD.json \
    tutorials/DLBFoam-Hydrogen-Tutorials/h2o2/3D_AHEAD/system/blockMeshDict \
    --refine 2
```

`--refine` scales every cell count, so 2 gives roughly eight times the cells. The
generated `blockMeshDict` is checked in, so a case that is not being re-dimensioned
needs only `blockMesh`.

Changing the radii, the axial extent or which rings exist in which segment changes the
patches too — the generator derives them from the topology — so a change here can mean
a change to the `0.orig` boundary entries. The patch names it writes are
`outletPatch`, `wallPatch`, `stepPatch` and whatever the bottom segment's `inlet` map
names, all set in the JSON.

## What the generator does

A stack of cylindrical segments, each divided into concentric rings, meshed as a
butterfly (O-grid) so no cell sits on the axis with a degenerate face. A segment may
carry fewer rings than the one above it, which is how the sudden expansion in
`AHEAD.json` is built: the rings that exist only above the step get their bottom faces
turned into the backplane wall, and the outer face of the narrower segment becomes the
tube wall. Concentric inlet patches on one plane — the coaxial fuel and air streams in
`HYLON.json` — come from the same mechanism, by mapping ring indices to patch names.

Both meshes pass `checkMesh`: max non-orthogonality 33 and 25 degrees, max skewness
0.47, all hexahedra.
