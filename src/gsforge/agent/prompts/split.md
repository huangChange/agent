## Stage: instance split

One segmented gaussian group can contain several physical objects (a table with a remote on it). They must be delivered as separate instances, and the support surface under each removed object must be completed.

- split_instances proposes instances from geometry (support plane + connectivity). Check its proposal against the gaussian previews: one instance should be one rigid object (or one articulated object, such as a cabinet with drawers, which stays a single instance).
- Over-split: a part of one object separated (a chair back split from its seat). Under-split: two objects touching each other kept together.
- Use patch_support_surface on the support instance so the area under removed objects is closed and colored consistently.
