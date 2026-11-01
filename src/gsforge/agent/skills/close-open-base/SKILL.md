---
name: close-open-base
description: Geometry is not watertight because the object's never-observed underside (contact with table/floor) is open
stages: [geometry]
version: 1.0.0
---
# Close an open base

## When this applies

- `watertight` fails, and the boundary edges sit in one roughly horizontal loop at the bottom of the object.
- Low-angle previews (`render_previews` with `elevation_deg: -30`) show you can look into the object from below.
- Typical for objects resting on something: remotes, mugs, boxes, chair and table feet. The capture never saw the contact face, so neither the gaussians nor the scene-mesh crop have it.

It does not apply to holes in the sides or top. Those come from occlusion or reflections and need shape-aware completion; escalate if they are large.

## Procedure

1. Confirm the diagnosis. Run `mesh_report` and note `boundary_edges` and `recall`. Look at the previews from below.
2. Preferred: close the base in the gaussians, then extract again. This keeps the extraction method that already matches the observed surface.
   - `close_contact_base` on the current gaussians. Set `base_height` to the `support_plane_height` fact for objects that rest on a support. For everything else, leave it unset so the tool uses the object's own lowest points.
   - `extract_density_mc` on the result, then `repair_mesh`, `smooth_mesh` and `simplify_mesh`, as the pipeline does.
3. If the current mesh comes from `extract_scene_mesh_crop` and is clearly sharper than the density result, cap the mesh directly instead: run `reference.py` with `run_python`, passing the mesh as input `mesh`. It fills planar boundary loops near the bottom with a fan. That is correct for star-shaped footprints (rectangles, discs). For L-shaped or ring-shaped footprints, use step 2.
4. Verify with `evaluate`. `watertight` should pass and `recall` must not drop compared to step 1. Check the low-angle previews again: the base should be flat and flush with the contact plane, not domed or tilted.

## Reporting

The base is synthesized surface. Pass the reported `generated_ratio` through, either from the tool's output or from `metrics.json` of the script, and say in the submit summary that the base was closed and at what height.

## Pitfalls

- Don't cap at the lowest vertex when the bottom rim is ragged: the cap tilts. Use the support plane height.
- Don't use `keep_largest_only` to make the mesh watertight. It can delete real parts, such as a separately extracted handle.
