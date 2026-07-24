#!/usr/bin/env python3
"""Regrid an Elmer VTU/PVTU output onto a structured grid and write NetCDF,
for use as an initial condition on a different mesh (e.g. one with different
resolution or centerline refinement) via Elmer/Ice's GridDataReader solver.

Accepts either a .pvtu (parallel, reads the referenced .vtu pieces
automatically) or a plain serial .vtu.

Two modes:
  --list-variables    Print available point-data fields and the detected
                       x/y bounding box, then exit without writing anything.
  --output PATH        Regrid the selected --variables (default: all) onto a
                       structured grid and write them to a NetCDF file.

Grid construction: the x/y extent is taken from the mesh's own bounding box.
If --dx (and optionally --dy) is not given, the domain is split into
--ncells (default 100) cells per axis, dividing it exactly. If --dx/--dy is
given and does not divide the extent exactly, the number of cells is rounded
up and the leftover is split evenly as padding on both sides, so the output
grid fully encompasses the mesh (--dy defaults to --dx, i.e. square cells,
if only --dx is given).

Points on the padded/output grid that fall outside the source mesh are
filled by nearest-neighbor extrapolation from the nearest valid grid point.

Vector fields (e.g. Elmer's 2-DOF SSAVelocity, which VTK output pads to 3
components with an all-zero third) are split into separate scalar NetCDF
variables per component, named "<field>_1", "<field>_2", ... matching the
convention used to reference vector components in a sif (e.g. "SSAVelocity
1"). An all-zero trailing third component is dropped.

BP/FS output is a 3D (extruded) volumetric mesh, not the inherently-2D mesh
SSA writes, so there is no single well-defined x/y surface to regrid unless
one is extracted first. If the input mesh contains volumetric cells, a
single boundary is extracted (by Elmer's "GeometryIds" cell-data field,
written when "Save Geometry Ids = Logical True" is set on the output
solver) before regridding. Use --surface-id to choose which one; if not
given on a volumetric mesh it defaults to 104 (Elmer's convention for
Boundary Condition 4, which is "lower_surface" in this project family's
BP/FS sifs).
"""
import argparse

from elmerice_pytools.vtu_mesh import (
    DEFAULT_SURFACE_ID, extract_surface_by_id, is_volumetric, load_vtu)
from elmerice_pytools.regrid import regrid_to_structured_grid
from elmerice_pytools.netcdf_io import print_griddata_sif_snippet, write_structured_netcdf


def list_variables(mesh):
    xmin, xmax, ymin, ymax, _, _ = mesh.bounds
    print(f"bounding box: x in [{xmin}, {xmax}], y in [{ymin}, {ymax}]")
    print(f"n points: {mesh.n_points}")
    print("point data fields:")
    for name in mesh.point_data.keys():
        arr = mesh.point_data[name]
        ncomp = arr.shape[1] if arr.ndim > 1 else 1
        print(f"  {name!r}: {ncomp} component(s), dtype={arr.dtype}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="input .pvtu or .vtu file")
    p.add_argument("--list-variables", action="store_true",
                    help="list available point-data fields and the bounding "
                         "box, then exit without writing anything")
    p.add_argument("--variables", nargs="+", default=None,
                    help="point-data fields to regrid (default: all)")
    p.add_argument("--output", "-o", help="output NetCDF file path")
    p.add_argument("--dx", type=float, default=None,
                    help="grid cell size in x, m (default: split extent into "
                         "--ncells cells)")
    p.add_argument("--dy", type=float, default=None,
                    help="grid cell size in y, m (default: same as --dx if "
                         "given, else split extent into --ncells cells)")
    p.add_argument("--ncells", type=int, default=100,
                    help="default cells per axis when --dx/--dy are not "
                         "given (default 100)")
    p.add_argument("--x-name", default="x", help="output x dimension/variable name")
    p.add_argument("--y-name", default="y", help="output y dimension/variable name")
    p.add_argument("--surface-id", type=int, default=None,
                    help="Elmer GeometryIds value of the boundary to extract "
                         "before regridding (100 + Boundary Condition number, "
                         "e.g. 104 for Boundary Condition 4). Only relevant "
                         "for volumetric (BP/FS) output; ignored (with a "
                         "warning) if given for an already-2D mesh. Defaults "
                         f"to {DEFAULT_SURFACE_ID} on a volumetric mesh if "
                         "not given.")
    return p.parse_args()


def main():
    args = parse_args()
    mesh = load_vtu(args.input)

    volumetric = is_volumetric(mesh)
    if args.surface_id is not None:
        if not volumetric:
            print(f"warning: --surface-id {args.surface_id} given but input "
                  "mesh has no volumetric cells (already a 2D surface) - "
                  "ignoring and using the whole mesh")
        else:
            mesh = extract_surface_by_id(mesh, args.surface_id)
    elif volumetric:
        print(f"input is a volumetric (3D) mesh; no --surface-id given, "
              f"defaulting to {DEFAULT_SURFACE_ID} (see --help)")
        mesh = extract_surface_by_id(mesh, DEFAULT_SURFACE_ID)

    if args.list_variables:
        list_variables(mesh)
        return

    if not args.output:
        raise SystemExit("--output is required unless --list-variables is given")

    variables = args.variables if args.variables else list(mesh.point_data.keys())
    x, y, fields = regrid_to_structured_grid(mesh, variables, args.dx, args.dy, args.ncells)
    write_structured_netcdf(args.output, x, y, fields, args.x_name, args.y_name)

    xmin, xmax, ymin, ymax, _, _ = mesh.bounds
    print(f"source bounds: x in [{xmin}, {xmax}], y in [{ymin}, {ymax}]")
    print(f"output grid: {len(x)} x {len(y)} points, "
          f"x in [{x[0]}, {x[-1]}], y in [{y[0]}, {y[-1]}]")
    print(f"wrote {args.output}")
    print_griddata_sif_snippet(fields, args.x_name, args.y_name, args.output)


if __name__ == "__main__":
    main()
