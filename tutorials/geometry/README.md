# Burner geometries

Neither 3D burner's `blockMeshDict` is written by hand. Both are generated from
the JSON here by
[utilities/cylindricalBurnerBlockMesh.py](../../../utilities/cylindricalBurnerBlockMesh.py),
which lays a stack of cylindrical segments out as a butterfly (O-grid) so that
no cell sits on the axis with a degenerate face. All hexahedra; both meshes pass
`checkMesh` with max non-orthogonality 23 degrees and max skewness 0.43.

| File | Burner | Cells |
|---|---|---|
| [nonPremixedBurner.json](nonPremixedBurner.json) | coaxial injector: central fuel bore, annular air, backplane | 39 936 |
| [premixedBurner.json](premixedBurner.json) | mixing tube discharging through a sudden expansion | 39 924 |

Read the `_comment` block at the top of each file before comparing anything to a
published number: it states which dimensions are taken from a case description
and which are assumed.

## Regenerating

```bash
../../utilities/cylindricalBurnerBlockMesh.py \
    nonPremixedBurner.json \
    ../3D_nonPremixed_burner/system/blockMeshDict
```

`--refine 2` scales every cell count, so it gives roughly eight times the cells.
The counts checked in put each burner at about 40 000 cells, a few times the
thread count of the GPU the chemistry model runs on -- see the mesh-size note in
[../README.md](../README.md) for why that, and not something cheaper.

The generator derives the patches from the topology, so changing the radii, the
axial extent or which rings exist in which segment can change the patch names --
and then the boundary entries of every `0.<mechanism>` have to change with them. The names it
writes are `outletPatch`, `wallPatch`, `stepPatch` and whatever the bottom
segment's `inlet` map names, all set in the JSON.
