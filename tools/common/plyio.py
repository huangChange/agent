"""Minimal PLY reader/writer (numpy only).

Supports ascii / binary_little_endian / binary_big_endian, scalar properties and
list properties (fast path for faces where every row has the same length).
"""
from __future__ import annotations


import numpy as np

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_NP_TO_PLY = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
              "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}


def _read_header(f):
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt = None
    elements = []
    comments = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected end of file inside PLY header")
        parts = line.decode("ascii", "replace").strip().split()
        if not parts:
            continue
        key = parts[0]
        if key == "format":
            fmt = parts[1]
        elif key == "comment":
            comments.append(" ".join(parts[1:]))
        elif key == "element":
            elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
        elif key == "property":
            if parts[1] == "list":
                # ("list", name, count_type, item_type)
                elements[-1]["props"].append(("list", parts[4], parts[2], parts[3]))
            else:
                elements[-1]["props"].append(("scalar", parts[2], parts[1]))
        elif key == "end_header":
            break
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"unsupported PLY format: {fmt}")
    return fmt, elements, comments


def _read_binary_element(f, el, endian):
    props, n = el["props"], el["count"]
    if all(p[0] == "scalar" for p in props):
        dt = np.dtype([(p[1], endian + _TYPES[p[2]]) for p in props])
        arr = np.fromfile(f, dtype=dt, count=n)
        if len(arr) != n:
            raise ValueError(f"element '{el['name']}' truncated")
        return {p[1]: arr[p[1]] for p in props}

    # Fast path: a single list property with constant row length (triangle faces).
    if len(props) == 1 and n > 0:
        _, name, ctype, itype = props[0]
        cdt = np.dtype(endian + _TYPES[ctype])
        idt = np.dtype(endian + _TYPES[itype])
        pos = f.tell()
        k = int(np.frombuffer(f.read(cdt.itemsize), dtype=cdt)[0])
        f.seek(pos)
        dt = np.dtype([("n", cdt), ("v", idt, (k,))])
        arr = np.fromfile(f, dtype=dt, count=n)
        if len(arr) == n and np.all(arr["n"] == k):
            return {name: arr["v"].astype(np.int64)}
        f.seek(pos)

    # Generic slow path.
    out = {p[1]: [] for p in props}
    for _ in range(n):
        for p in props:
            if p[0] == "scalar":
                dt = np.dtype(endian + _TYPES[p[2]])
                out[p[1]].append(np.frombuffer(f.read(dt.itemsize), dtype=dt)[0])
            else:
                cdt = np.dtype(endian + _TYPES[p[2]])
                idt = np.dtype(endian + _TYPES[p[3]])
                k = int(np.frombuffer(f.read(cdt.itemsize), dtype=cdt)[0])
                out[p[1]].append(np.frombuffer(f.read(idt.itemsize * k), dtype=idt).copy())
    for p in props:
        if p[0] == "scalar":
            out[p[1]] = np.asarray(out[p[1]])
    return out


def _read_ascii(f, elements):
    data = {}
    for el in elements:
        props, n = el["props"], el["count"]
        rows = [f.readline().split() for _ in range(n)]
        if all(p[0] == "scalar" for p in props):
            arr = np.asarray(rows, dtype=np.float64).reshape(n, len(props))
            data[el["name"]] = {p[1]: arr[:, i].astype(_TYPES[p[2]]) for i, p in enumerate(props)}
            continue
        out = {p[1]: [] for p in props}
        for row in rows:
            i = 0
            for p in props:
                if p[0] == "scalar":
                    out[p[1]].append(float(row[i]))
                    i += 1
                else:
                    k = int(row[i])
                    out[p[1]].append(np.asarray(row[i + 1:i + 1 + k], dtype=np.int64))
                    i += 1 + k
        for p in props:
            if p[0] == "scalar":
                out[p[1]] = np.asarray(out[p[1]], dtype=_TYPES[p[2]])
            else:
                lens = {len(x) for x in out[p[1]]}
                if len(lens) == 1:
                    out[p[1]] = np.stack(out[p[1]]) if out[p[1]] else np.zeros((0, 3), np.int64)
        data[el["name"]] = out
    return data


def read_ply(path):
    """Return {'format', 'elements' (header spec), 'comments', 'data': {element: {prop: array}}}."""
    with open(path, "rb") as f:
        fmt, elements, comments = _read_header(f)
        if fmt == "ascii":
            data = _read_ascii(f, elements)
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            data = {el["name"]: _read_binary_element(f, el, endian) for el in elements}
    return {"format": fmt, "elements": elements, "comments": comments, "data": data}


def read_ply_header(path):
    with open(path, "rb") as f:
        fmt, elements, comments = _read_header(f)
    return {"format": fmt, "elements": elements, "comments": comments}


def write_ply(path, vertex: dict, faces=None):
    """Write binary little-endian PLY. `vertex` maps property name -> 1D array (order kept)."""
    names = list(vertex.keys())
    n = len(vertex[names[0]])
    cols = []
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for name in names:
        a = np.asarray(vertex[name])
        code = a.dtype.str[1:]
        if code not in _NP_TO_PLY:
            a = a.astype(np.float32)
            code = "f4"
        cols.append((name, "<" + code, a))
        header.append(f"property {_NP_TO_PLY[code]} {name}")
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int32)
        header += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    header.append("end_header")
    arr = np.empty(n, dtype=[(c[0], c[1]) for c in cols])
    for name, _, a in cols:
        arr[name] = a
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        arr.tofile(f)
        if faces is not None:
            fa = np.empty(len(faces), dtype=[("n", "u1"), ("v", "<i4", (faces.shape[1],))])
            fa["n"] = faces.shape[1]
            fa["v"] = faces
            fa.tofile(f)


__all__ = ["read_ply", "read_ply_header", "write_ply"]
