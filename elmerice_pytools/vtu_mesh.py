"""Loading, cropping, surface-extraction and point-sampling helpers for
Elmer VTU/PVTU output, shared between projects (originally split out of
BlIP's tools/pvtu_to_netcdf.py and Fimbul's tools/extract_subdomain.py,
which had each grown their own copies of the same z-flattening/sampling
logic independently)."""
import numpy as np
import pyvista as pv
from scipy.interpolate import NearestNDInterpolator

# Elmer's default BC Id Offset (fem/src/modules/ResultOutputSolve/
# VtuOutputSolver.F90): a boundary's GeometryIds value is 100 + <its
# Boundary Condition number in the sif>. 104 = Boundary Condition 4 =
# "lower_surface" in this project family's BP/FS sifs.
DEFAULT_SURFACE_ID = 104

# VTK cell type codes for volumetric (3D) cells likely to appear in an Elmer
# extruded mesh (linear and quadratic tetrahedra, hexahedra, wedges,
# pyramids). Used only to decide whether surface extraction is needed.
VOLUMETRIC_CELL_TYPES = {10, 12, 13, 14, 24, 25, 26, 27, 29, 30, 31, 32}

VTK_TRIANGLE = 5


def load_vtu(path):
    """Plain read of a .pvtu or .vtu file -- no assumptions about welding
    or cell types, unlike Fimbul's load_triangles (which additionally
    assumes/handles a specific partitioned-mesh-merge quirk); compose with
    weld_duplicate_nodes/extract_triangles below if that's needed."""
    return pv.read(path)


def weld_duplicate_nodes(mesh, tolerance=1e-6):
    """Weld coincident nodes. Needed e.g. when a .pvtu is a parallel
    decomposition merged back together, where each partition piece
    contributes its own copy of the nodes along its boundary with its
    neighbors -- the merged mesh is topologically disconnected into pieces
    despite being geometrically seamless, which breaks anything relying on
    cell-adjacency (connected-component filtering, boundary-edge finding)."""
    n_before = mesh.n_points
    welded = mesh.clean(tolerance=tolerance, produce_merge_map=False)
    print(f"welded {n_before - welded.n_points} duplicate nodes "
          f"({n_before} -> {welded.n_points} points)")
    return welded


def extract_triangles(mesh):
    """Keep only 2D triangle cells, dropping e.g. a source mesh's own 1D
    boundary LINE cells."""
    tri_mask = mesh.celltypes == VTK_TRIANGLE
    return mesh.extract_cells(np.where(tri_mask)[0])


def is_volumetric(mesh):
    """True if the mesh contains any 3D (volumetric) cells -- i.e. this is
    an extruded BP/FS mesh, not an inherently-2D SSA one."""
    return any(int(t) in VOLUMETRIC_CELL_TYPES for t in mesh.celltypes)


def extract_surface_by_id(mesh, geometry_id):
    """Return the subset of mesh whose "GeometryIds" cell-data field equals
    geometry_id (an Elmer boundary, e.g. 104 for Boundary Condition 4).
    Needed before regridding/sampling a BP/FS volumetric mesh: there's no
    single well-defined x/y surface to sample from a 3D extruded mesh
    otherwise. Re-run the Elmer output solver with "Save Geometry Ids =
    Logical True" if this field is missing."""
    if "GeometryIds" not in mesh.cell_data:
        raise KeyError(
            "Mesh has no 'GeometryIds' cell data - re-run the Elmer output "
            "solver with 'Save Geometry Ids = Logical True', or pass "
            "geometry_id only when that is available.")
    ids = mesh.cell_data["GeometryIds"]
    mask = ids == geometry_id
    if not np.any(mask):
        available = sorted(set(int(i) for i in ids))
        raise ValueError(
            f"No cells found with GeometryIds == {geometry_id}. "
            f"Available GeometryIds in this file: {available}")
    surface = mesh.extract_cells(mask)
    print(f"extracted surface GeometryIds={geometry_id}: "
          f"{surface.n_points} points, {surface.n_cells} cells")
    return surface


def crop_mesh_to_bbox(mesh, xmin, xmax, ymin, ymax):
    """Crop to an x/y bounding box (z left unbounded) -- for cutting a
    whole-domain mesh down to a small area of interest before a regrid or
    sample step, so that step isn't spent on geometry far outside where it's
    actually needed."""
    zmin, zmax = mesh.bounds[4], mesh.bounds[5]
    pad = max(abs(zmin), abs(zmax), 1.0)
    return mesh.clip_box((xmin, xmax, ymin, ymax, zmin - pad, zmax + pad),
                          invert=False)


def sanitize_name(name):
    return name.strip().replace(" ", "_")


def split_vector_components(name, arr):
    """Yield (output_name, 1D array) pairs for a point-data field, splitting
    vector fields into scalar components and dropping an all-zero padded
    third component (VTK pads e.g. Elmer's 2-DOF SSAVelocity to 3 with an
    all-zero third)."""
    if arr.ndim == 1:
        yield sanitize_name(name), arr
        return
    ncomp = arr.shape[1]
    if ncomp == 3 and np.all(arr[:, 2] == 0):
        ncomp = 2
    for i in range(ncomp):
        yield f"{sanitize_name(name)}_{i + 1}", arr[:, i]


def _flatten_z(mesh):
    """Copy of mesh with z zeroed. .sample()/probing does a real 3D
    point-in-cell test, so a query point/grid fixed at z=0 mostly misses a
    mesh whose actual z (e.g. a BP/FS surface's zb/zs, often hundreds of
    metres from 0) isn't ~0 everywhere -- almost every point then falls back
    to nearest-neighbor fill. We only want x/y interpolation here (z is
    *data*, e.g. one of the fields being sampled, not geometry that should
    matter for it), so probe against a flattened copy instead."""
    flat = mesh.copy()
    flat.points[:, 2] = 0.0
    return flat


def sample_fields_at_points(source_mesh, xy, field_names):
    """Sample point-data fields from source_mesh at arbitrary (N, 2) xy
    locations (e.g. a different mesh's own nodes, or a structured grid
    flattened to a point list) via VTK point-location + interpolation.
    Points falling outside source_mesh are filled by nearest-neighbor
    extrapolation from the nearest valid sample, rather than left at VTK's
    raw (meaningless, typically 0) value.

    Returns {field_name: array}, splitting vector fields into "<name>_1",
    "<name>_2", ... per split_vector_components. All requested fields are
    sampled off one shared probe/validity pass, not one pass per field.
    """
    probe = pv.PolyData(np.column_stack([xy, np.zeros(len(xy))]))
    flat = _flatten_z(source_mesh)
    result = probe.sample(flat)
    valid = np.asarray(result["vtkValidPointMask"]).astype(bool)

    out = {}
    for name in field_names:
        if name not in source_mesh.point_data:
            raise KeyError(f"{name!r} not found in point data; available: "
                            f"{list(source_mesh.point_data.keys())}")
        for out_name, values in split_vector_components(name, result[name]):
            if not np.all(valid):
                filler = NearestNDInterpolator(xy[valid], values[valid])
                values = values.copy()
                values[~valid] = filler(xy[~valid])
            out[out_name] = values
    return out
