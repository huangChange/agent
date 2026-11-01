## Stage: texture

The goal is a base-color texture that looks like the captured object under neutral lighting, covers the whole surface, and keeps text, logos and fine patterns legible.

- Capture photos, when available, are much sharper than gaussian colors. Prefer bake_from_capture; use vertex colors from gaussians only as a fallback.
- generated_ratio is the share of the surface no observation covered. Fill it so the object does not look half-finished: continue materials and patterns from adjacent observed regions, respect symmetry, and keep seams invisible. Plain diffusion fill is acceptable only for small regions or uniform materials.
- Blurry text or labels: if the text is readable in any capture photo, reconstruct it from there. If you can infer it with confidence (a remote's button labels, a brand mark that is clearly visible elsewhere on the object), you may restore it, and you must say so in the summary. Do not invent text that no view supports.
- Baked lighting (hard shadows, specular highlights burned into the albedo) makes assets look wrong under simulator lighting. Mention it in the summary if you see it and cannot remove it.
