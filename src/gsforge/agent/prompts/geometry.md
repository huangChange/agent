## Stage: geometry

The goal is a mesh that (a) agrees with the gaussians in both directions (recall: the mesh covers what was observed; precision: the mesh doesn't add surface that wasn't there), and (b) is a clean closed 2-manifold so simulators can compute volume, mass and contacts.

Typical failure patterns and what usually causes them:
- Low recall on thin parts (legs, handles, edges): extraction resolution too low or iso level too high. Try extract_density_mc with higher resolution or lower level_ratio, or the scene mesh crop if it captured the part.
- Low precision / bloated surface: iso level too low, floaters not cleaned, or the scene mesh crop picked up a neighbouring object. Check previews for blobs.
- Not watertight / boundary edges: open contact regions (the underside where an object rested) or holes from occlusion. repair_mesh closes small holes; large openings are unobserved surface and need completion that follows the object's shape (for example extend a planar base, mirror a symmetric part).
- Many components: floaters or a merged neighbour. Inspect before deleting: a small separate component can be a real part (a knob).
- Non-manifold edges: usually from crops or concatenated parts; repair_mesh or re-extraction fixes them.

Different candidates fail differently. It is often best to start from the candidate with the highest recall and fix its topology, rather than to fix recall of a clean but incomplete mesh.
