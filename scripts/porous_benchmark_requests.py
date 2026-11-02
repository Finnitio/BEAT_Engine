"""Write the Air, JCA, JCAL and JCA+limp variants of one captured request for the porous benchmark.

The porous body is one bounded region of a request captured with
capture_boundary_lab_request.py. Every variant solves the same mesh; only the
body's `loss_model` changes. The Air variant keeps the captured loss model, so
it is exactly what the project solves today.

  python scripts/porous_benchmark_requests.py sawmod.json --body region:sealedmainchamber --out-dir porous
  python scripts/porous_benchmark_requests.py sealed.json --body region:interior-air-1 \\
      --split-below-m -0.2 --out-dir porous

--split-below-m turns the tetrahedra of the body whose centroid lies below that
coordinate (metres, along --split-axis) into a new region on the same mesh, the
porous layer; the shared faces carry no triangle, so the two regions meet at a
region seam. The rest of the body stays air. The split mesh is sent inline
(`mesh_data`), so this needs meshio and numpy: run it in the Boundary Lab
environment, like the capture script. Only rigid boundaries may touch the layer;
their layer triangles get a new tag and a rigid boundary of the layer region.

The material is the soft fibrous material of Allard & Atalla 2nd ed. Table 11.2
(the engine's filled-tube tests): JCA rigid, JCAL with k0' = 2e-9 m^2 (half of
JCA's phi Lambda'^2/8) and JCA on a limp frame with rho1 = 30 kg/m^3.
"""

import argparse
import base64
import copy
import json
from pathlib import Path

SOFT_FIBROUS = {
    "model": "jca",
    "frame": "rigid",
    "porosity": 0.98,
    "flow_resistivity_pa_s_per_m2": 25000.0,
    "tortuosity": 1.02,
    "viscous_characteristic_length_m": 90e-6,
    "thermal_characteristic_length_m": 180e-6,
}
MATERIALS = {
    "jca": SOFT_FIBROUS,
    "jcal": {**SOFT_FIBROUS, "model": "jcal", "static_thermal_permeability_m2": 2e-9},
    "limp": {**SOFT_FIBROUS, "frame": "limp", "frame_density_kg_per_m3": 30.0},
}


def packed(array, dtype):
    import numpy as np

    data = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return {"dtype": dtype, "shape": list(data.shape), "data": base64.b64encode(data.tobytes()).decode()}


def split_body(request, body_id, axis, below_m):
    """Split the body's tetrahedra at `below_m` into a seam-joined layer region; returns its id."""
    import meshio
    import numpy as np

    system = request["compiled_system"]
    body = next(r for r in system["regions"] if r["id"] == body_id)
    (mesh_id,) = body["mesh_ids"]
    resource = next(m for m in system["meshes"] if m["id"] == mesh_id)
    if any(mesh_id in json.dumps(i) for i in system.get("interfaces", [])):
        raise SystemExit(f"{mesh_id} carries interfaces; splitting would renumber their vertices")
    mesh = meshio.read(resource["file"])
    points = mesh.points * resource["scale_to_m"] + np.asarray(resource["translation_m"])
    volume_tags = {g["tag"] for g in body["volume_groups"]}
    tags = mesh.cell_data["gmsh:physical"]
    blocks = [(c.type, c.data, np.asarray(t)) for c, t in zip(mesh.cells, tags)]
    if any(kind not in ("triangle", "tetra") for kind, _, _ in blocks):
        raise SystemExit("only P1 meshes (triangle, tetra) are supported")
    names = {name: [int(v[0]), int(v[1])] for name, v in mesh.field_data.items()}
    next_tag = max(v[0] for v in names.values()) + 1
    layer_volume_tag = next_tag

    cells = []
    layer_faces = set()
    for kind, conn, tag in blocks:
        if kind != "tetra":
            continue
        centroid = points[conn].mean(axis=1)[:, "xyz".index(axis)]
        in_layer = np.isin(tag, list(volume_tags)) & (centroid < below_m)
        if not in_layer.any() or in_layer.all():
            raise SystemExit("the split leaves the body or the layer empty")
        tag = np.where(in_layer, layer_volume_tag, tag)
        for tet in conn[in_layer]:
            for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                layer_faces.add(tuple(sorted(tet[list(face)])))
        cells.append(("tetra", conn, tag))
    names["PorousLayer"] = [layer_volume_tag, 3]

    boundaries = {b["group"]["tag"]: b for b in system["boundaries"] if b["region_id"] == body_id}
    layer_tag_for = {}
    triangles = []
    for kind, conn, tag in blocks:
        if kind != "triangle":
            continue
        tag = tag.copy()
        for row, triangle in enumerate(conn):
            if tuple(sorted(triangle)) not in layer_faces or tag[row] not in boundaries:
                continue
            boundary = boundaries[tag[row]]
            if boundary["kind"] != "rigid":
                raise SystemExit(f"{boundary['id']} ({boundary['kind']}) touches the layer; move the split")
            if tag[row] not in layer_tag_for:
                next_tag += 1
                layer_tag_for[tag[row]] = next_tag
                names[f"PorousLayer {boundary['name']}"] = [next_tag, 2]
            tag[row] = layer_tag_for[tag[row]]
        triangles.append(("triangle", conn, tag))

    resource.update(file="", scale_to_m=1.0, translation_m=[0.0, 0.0, 0.0])
    resource["mesh_data"] = {
        "schema_version": 1,
        "points": packed(points, "<f8"),
        "cells": [
            {"type": kind, "connectivity": packed(conn, "<i8"), "physical_tags": packed(tag, "<i8")}
            for kind, conn, tag in triangles + cells
        ],
        "physical_names": names,
    }
    layer_id = f"{body_id}-porous-layer"
    layer = copy.deepcopy(body)
    layer.update(
        id=layer_id,
        name=f"{body['name']} porous layer",
        volume_groups=[{"mesh_id": mesh_id, "dimension": 3, "tag": layer_volume_tag, "name": "PorousLayer"}],
    )
    system["regions"].append(layer)
    for original, new in layer_tag_for.items():
        name = f"PorousLayer {boundaries[original]['name']}"
        system["boundaries"].append(
            {
                "id": f"boundary:{layer_id}:{new}",
                "name": name,
                "region_id": layer_id,
                "kind": "rigid",
                "parameters": {},
                "group": {"mesh_id": mesh_id, "dimension": 2, "tag": int(new), "name": name},
            }
        )
    layer_tets = sum(int((tag == layer_volume_tag).sum()) for _, _, tag in cells)
    print(f"split {body_id}: {layer_tets} tetrahedra below {axis} = {below_m} m form {layer_id}")
    return layer_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", type=Path, help="request captured with capture_boundary_lab_request.py")
    parser.add_argument("--body", required=True, help="id of the bounded region that becomes the porous body")
    parser.add_argument("--split-below-m", type=float, help="split the body here into air + a porous layer")
    parser.add_argument("--split-axis", default="z", choices=("x", "y", "z"))
    parser.add_argument(
        "--phasor-convention",
        default="exp(+i omega t)",
        help="solver_options.phasor_convention; Boundary Lab solves send exp(+i omega t) (default)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text())
    request.setdefault("solver_options", {})["phasor_convention"] = args.phasor_convention
    body = args.body
    if args.split_below_m is not None:
        body = split_body(request, args.body, args.split_axis, args.split_below_m)
    region = next((r for r in request["compiled_system"]["regions"] if r["id"] == body), None)
    if region is None or region["kind"] != "bounded_air":
        raise SystemExit(f"{body} is not a bounded region of {args.request}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.request.stem
    for name, material in [("air", region["loss_model"]), *MATERIALS.items()]:
        variant = copy.deepcopy(request)
        target = next(r for r in variant["compiled_system"]["regions"] if r["id"] == body)
        target["loss_model"] = copy.deepcopy(material)
        out = args.out_dir / f"{stem}_{name}.json"
        out.write_text(json.dumps(variant))
        print(f"{out}: {body} loss_model {json.dumps(material)}")


if __name__ == "__main__":
    main()
