"""Cap open, roughly planar boundary loops at the bottom of a mesh (sandbox script for run_python).

Input:   inputs/mesh.ply (or any mesh format trimesh reads), +Z up
Output:  outputs/capped.ply, outputs/metrics.json {watertight, generated_ratio, loops_capped}
Correct for star-shaped loops (rectangles, discs); other footprints need close_contact_base instead.
"""

import json

import numpy as np
import trimesh

PLANARITY = 0.02  # max height spread of a loop, as a fraction of the object height
BOTTOM_BAND = 0.15  # loops must lie within this fraction of the height above the lowest point

with open("inputs/manifest.json") as f:
    manifest = json.load(f)
mesh = trimesh.load(manifest["mesh"]["path"], force="mesh")
mesh.merge_vertices()
height = float(np.ptp(mesh.vertices[:, 2])) or 1.0
z_min = float(mesh.vertices[:, 2].min())

# boundary edges = edges used by exactly one face; chain them into closed loops
edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
boundary = edges[counts == 1]
neighbours: dict[int, list[int]] = {}
for a, b in boundary:
    neighbours.setdefault(int(a), []).append(int(b))
    neighbours.setdefault(int(b), []).append(int(a))

loops, seen = [], set()
for start in neighbours:
    if start in seen or len(neighbours[start]) != 2:
        continue
    loop, prev, cur = [start], None, start
    while True:
        seen.add(cur)
        nxt = [n for n in neighbours[cur] if n != prev and len(neighbours[n]) == 2]
        if not nxt or nxt[0] == start:
            break
        prev, cur = cur, nxt[0]
        if cur in seen:
            break
        loop.append(cur)
    if len(loop) >= 3 and start in neighbours[loop[-1]]:
        loops.append(loop)

new_faces, cap_area, capped = [], 0.0, 0
vertices = mesh.vertices.copy()
for loop in loops:
    z = vertices[loop, 2]
    if np.ptp(z) > PLANARITY * height or z.mean() > z_min + BOTTOM_BAND * height:
        print(f"skip loop of {len(loop)} vertices (not a planar bottom loop)")
        continue
    center = vertices[loop].mean(axis=0)
    c = len(vertices)
    vertices = np.vstack([vertices, center])
    for i in range(len(loop)):
        new_faces.append([loop[i], loop[(i + 1) % len(loop)], c])
    capped += 1

capped_mesh = trimesh.Trimesh(vertices, np.vstack([mesh.faces, np.array(new_faces, dtype=np.int64).reshape(-1, 3)]))
trimesh.repair.fix_winding(capped_mesh)
trimesh.repair.fix_inversion(capped_mesh, multibody=True)
if new_faces:
    cap_area = float(capped_mesh.area_faces[len(mesh.faces):].sum())
capped_mesh.export("outputs/capped.ply")
metrics = {"watertight": float(capped_mesh.is_watertight), "loops_capped": float(capped),
           "generated_ratio": cap_area / float(capped_mesh.area)}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f)
print(f"loops found {len(loops)}, capped {capped}, metrics {metrics}")
