"""Reading and writing Elmer's plain-text mesh format (mesh.nodes/
mesh.elements/mesh.boundary/mesh.header), and the boundary-edge topology
helpers built on top of it -- factored out of Fimbul's tools/
extract_subdomain.py so other projects (and other meshes within the same
project, e.g. a Gmsh-built coarse mesh converted via ElmerGrid) can reuse it
without depending on Fimbul's own streamline-tracing-specific code."""
import hashlib
import os

import numpy as np
import pyvista as pv

# Elmer element type codes (see ElmerSolver mesh format docs)
ELMER_LINE2 = 202
ELMER_TRIANGLE3 = 303


def mesh_fingerprint(mesh):
    """Short hash of node count + coordinates, so a mesh/ and a field
    netcdf generated in different runs (e.g. after re-extracting with
    different parameters) can be caught before they're fed to a node-order-
    matching reader with mismatched node correspondence -- that fails
    silently (wraps around) rather than erroring, so this is worth checking
    by hand if fields ever come out looking wrong in patches."""
    h = hashlib.sha256()
    h.update(np.asarray(mesh.n_points).tobytes())
    h.update(np.ascontiguousarray(mesh.points[:, :2]).tobytes())
    return h.hexdigest()[:16]


def load_elmer_mesh(mesh_dir):
    """Load an Elmer mesh (mesh.nodes/mesh.elements, e.g. from ElmerGrid
    converting a Gmsh-built mesh) as a pv.UnstructuredGrid of triangles --
    so find_boundary_edges/downstream boundary-classification code (written
    for meshes built directly in Python) also work on a mesh that came from
    elsewhere. Assumes node indices in mesh.nodes are the sequential 1..N
    ElmerGrid always writes, i.e. row order == node index order."""
    nodes = np.loadtxt(os.path.join(mesh_dir, "mesh.nodes"))
    pts = nodes[:, 2:5]
    elems = np.loadtxt(os.path.join(mesh_dir, "mesh.elements"),
                        usecols=(3, 4, 5), dtype=int) - 1
    faces = np.hstack([np.full((len(elems), 1), 3), elems]).ravel()
    return pv.PolyData(pts, faces).cast_to_unstructured_grid()


def find_boundary_edges(mesh):
    """Return dict: (local_node_i, local_node_j) -> parent (local) triangle
    index, for edges used by exactly one triangle. Keeps the edge's natural
    node order as it appears in the parent triangle's own (CCW) winding --
    NOT sorted by index. That matters once the mesh is extruded into 3D: a
    boundary edge becomes a quad face there, and an edge whose two nodes
    were reordered by an arbitrary index sort (independent of the parent
    triangle's winding) produces an inconsistently-wound, self-intersecting
    ("Degenerate") quad face -- roughly half of all boundary edges are
    affected, since sorting by index flips the order only when the
    higher-index node happens to come first in the natural winding.
    """
    cells = mesh.cells.reshape(-1, 4)[:, 1:4]
    edge_count = {}
    edge_parent = {}
    edge_order = {}
    for tri_idx, (i, j, k) in enumerate(cells):
        for a, b in ((i, j), (j, k), (k, i)):
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
            edge_parent[key] = tri_idx
            edge_order[key] = (a, b)
    return {edge_order[e]: edge_parent[e] for e, c in edge_count.items() if c == 1}


def write_elmer_mesh(mesh, outdir, body_id=1):
    os.makedirs(outdir, exist_ok=True)
    pts = mesh.points
    cells = mesh.cells.reshape(-1, 4)[:, 1:4]
    n_nodes = mesh.n_points
    n_bulk = len(cells)

    boundary_edges = find_boundary_edges(mesh)

    with open(os.path.join(outdir, "mesh.nodes"), "w") as f:
        for idx in range(n_nodes):
            x, y, z = pts[idx]
            f.write(f"{idx + 1} -1 {x:.6f} {y:.6f} {z:.6f}\n")

    with open(os.path.join(outdir, "mesh.elements"), "w") as f:
        for idx, (i, j, k) in enumerate(cells):
            f.write(f"{idx + 1} {body_id} {ELMER_TRIANGLE3} {i + 1} {j + 1} {k + 1}\n")

    fingerprint = mesh_fingerprint(mesh)
    with open(os.path.join(outdir, "fingerprint.txt"), "w") as f:
        f.write(fingerprint + "\n")
    print(f"mesh fingerprint: {fingerprint}  (must match the matching field "
          f"netcdf's node_fingerprint attribute)")

    return boundary_edges, n_nodes, n_bulk


def write_elmer_boundary_and_header(mesh, outdir, boundary_edges, labels, n_nodes, n_bulk):
    """labels: dict, same keys as boundary_edges, values are the integer
    Elmer boundary-condition ID to tag that edge with (matching whatever
    "Target Boundaries" a sif's Boundary Condition blocks use)."""
    with open(os.path.join(outdir, "mesh.boundary"), "w") as f:
        for b_idx, (edge, parent) in enumerate(boundary_edges.items()):
            i, j = edge
            bc = labels[edge]
            f.write(f"{b_idx + 1} {bc} {parent + 1} 0 {ELMER_LINE2} {i + 1} {j + 1}\n")
    n_boundary = len(boundary_edges)

    with open(os.path.join(outdir, "mesh.header"), "w") as f:
        f.write(f"{n_nodes} {n_bulk} {n_boundary}\n")
        f.write("2\n")
        f.write(f"{ELMER_LINE2} {n_boundary}\n")
        f.write(f"{ELMER_TRIANGLE3} {n_bulk}\n")

    return n_boundary
