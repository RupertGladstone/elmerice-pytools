#!/usr/bin/env python3
"""Sample an Elmer VTU/PVTU output's fields directly at another mesh's own
node (x, y) positions, and write a node-matching NetCDF for UGridDataReader.

For seeding one Elmer run's state from a *different* run's result when the
two meshes are dimensionally incompatible -- e.g. restarting a 2D SSA run
from a 3D BP/FS run's output. Elmer's own "Restart File" mechanism can't do
this: it matches values to the current mesh by node position in the
restart-writing mesh's own numbering, which requires the two meshes to
actually be the same mesh (same node count/order), not just overlapping in
space. UGridDataReader's "mesh =" option does genuine mesh-to-mesh
interpolation (InterpolateMeshToMesh) for cases where that's not true, but
needs the source mesh as an on-disk Elmer mesh directory (not produced by
Elmer's usual in-memory extrusion) and does a point-in-element search that
a flat 2D target (z=0) will rarely fall inside a 3D ice volume's real
elevation range for.

This sidesteps both problems by doing the interpolation here, in Python,
directly at the target mesh's own node coordinates -- the output is then
already in exact node-order correspondence with the target mesh (like this
project family's usual UGridDataReader-ready netcdf files), read back with
no "mesh =" and no ordering ambiguity.

BP/FS output is a 3D (extruded) volumetric mesh; a single boundary is
extracted first (see --surface-id) since a flat target has no well-defined
correspondence to a 3D volume otherwise. Fields that are logically
2D-footprint quantities exported across every extruded layer (h, zs, zb,
GroundedMask, a vertically-averaged velocity, ...) carry the same value at
every layer for a given (x, y), so which surface is extracted doesn't
matter for those -- it does matter for anything that is genuinely
layer-specific (e.g. the raw 3D "Velocity"), which this approach isn't
meant for.
"""
import argparse
import os

from elmerice_pytools.vtu_mesh import (
    DEFAULT_SURFACE_ID, extract_surface_by_id, is_volumetric, load_vtu,
    sample_fields_at_points)
from elmerice_pytools.elmer_mesh import load_elmer_mesh, mesh_fingerprint
from elmerice_pytools.netcdf_io import print_ugrid_sif_snippet, write_node_netcdf


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="source .pvtu or .vtu file to sample fields from")
    p.add_argument("--target-mesh", required=True,
                    help="Elmer mesh directory (mesh.nodes/mesh.elements) "
                         "whose own node positions to sample at")
    p.add_argument("--variables", nargs="+", required=True,
                    help="point-data fields to sample from the source")
    p.add_argument("--surface-id", type=int, default=None,
                    help="Elmer GeometryIds value of the boundary to extract "
                         "from the source before sampling (100 + Boundary "
                         "Condition number, e.g. 104 for Boundary Condition "
                         "4). Only relevant for a volumetric (BP/FS) source; "
                         "ignored (with a warning) if given for an "
                         f"already-2D source. Defaults to {DEFAULT_SURFACE_ID} "
                         "on a volumetric source if not given.")
    p.add_argument("--output", "-o", required=True, help="output NetCDF file path")
    return p.parse_args()


def main():
    args = parse_args()
    source = load_vtu(args.source)

    volumetric = is_volumetric(source)
    if args.surface_id is not None:
        if not volumetric:
            print(f"warning: --surface-id {args.surface_id} given but source "
                  "mesh has no volumetric cells (already a 2D surface) - "
                  "ignoring and using the whole mesh")
        else:
            source = extract_surface_by_id(source, args.surface_id)
    elif volumetric:
        print(f"source is a volumetric (3D) mesh; no --surface-id given, "
              f"defaulting to {DEFAULT_SURFACE_ID} (see --help)")
        source = extract_surface_by_id(source, DEFAULT_SURFACE_ID)

    target = load_elmer_mesh(args.target_mesh)
    xy = target.points[:, :2]

    fields = sample_fields_at_points(source, xy, args.variables)

    fingerprint = mesh_fingerprint(target)
    write_node_netcdf(target.n_points, fields, args.output, fingerprint=fingerprint)

    print(f"sampled {len(fields)} field(s) from {args.source} at "
          f"{target.n_points} node(s) of {args.target_mesh}")
    print(f"wrote {args.output}  (fingerprint {fingerprint}, matching "
          f"{os.path.join(args.target_mesh, 'fingerprint.txt')} if that "
          f"mesh was built by this project family's own tooling)")
    print_ugrid_sif_snippet(fields, args.output)


if __name__ == "__main__":
    main()
