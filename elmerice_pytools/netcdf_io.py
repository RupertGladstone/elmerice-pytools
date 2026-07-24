"""Writing the two netCDF shapes this project family reads back into Elmer:
one value per node in some target mesh's own node order (for
UGridDataReader's exact-match mode), or values on a regular x/y grid (for
GridDataReader's spatial-interpolation mode)."""
from netCDF4 import Dataset


def write_node_netcdf(n_points, fields, path, fingerprint=None, x=None, y=None):
    """fields: {name: (n_points,) array}. fingerprint, if given, is written
    as the node_fingerprint global attribute -- UGridDataReader itself
    doesn't check it, but it lets a mismatched mesh/field-file pairing (e.g.
    after re-extracting a mesh with different parameters) be caught by
    inspection instead of silently wrapping around to the wrong nodes."""
    with Dataset(path, "w") as nc:
        nc.n_nodes = n_points
        if fingerprint is not None:
            nc.node_fingerprint = fingerprint
        nc.createDimension("node", n_points)
        for name, values in fields.items():
            var = nc.createVariable(name.replace(" ", "_"), "f8", ("node",))
            var[:] = values
        if x is not None:
            xv = nc.createVariable("x", "f8", ("node",))
            xv[:] = x
        if y is not None:
            yv = nc.createVariable("y", "f8", ("node",))
            yv[:] = y


def write_structured_netcdf(path, x, y, fields, x_name="x", y_name="y"):
    """fields: {name: (len(y), len(x)) array}."""
    with Dataset(path, "w") as nc:
        nc.createDimension(x_name, len(x))
        nc.createDimension(y_name, len(y))
        xv = nc.createVariable(x_name, "f8", (x_name,))
        yv = nc.createVariable(y_name, "f8", (y_name,))
        xv[:] = x
        yv[:] = y
        for name, values in fields.items():
            var = nc.createVariable(name, "f8", (y_name, x_name))
            var[:, :] = values


def print_ugrid_sif_snippet(field_names, netcdf_path):
    print("\n--- Elmer sif snippet to read this back in (UGridDataReader, "
          "exact node match) ---")
    print("Solver N")
    print('  Exec Solver = "Before all"')
    print('  Equation = "UGridDataReader"')
    print('  Procedure = "ElmerIceSolvers" "UGridDataReader"')
    print(f'  File Name = File "{netcdf_path}"')
    for i, name in enumerate(field_names, start=1):
        print(f'  Variable Name {i} = File "{name}"')
        print(f'  Target Variable {i} = String "<elmer variable for {name}>"')
        print(f'  Exported Variable {i} = "<elmer variable for {name}>"')
    print("End\n")


def print_griddata_sif_snippet(field_names, x_name, y_name, netcdf_path):
    print("\n--- Elmer sif snippet to read this back in (GridDataReader, "
          "spatial interpolation) ---")
    print('Solver N')
    print('  Equation = "GridDataReader"')
    print('  Procedure = "GridDataReader" "GridDataReader"')
    print(f'  Filename = File "{netcdf_path}"')
    print(f'  X Dim Name = String "{x_name}"')
    print(f'  Y Dim Name = String "{y_name}"')
    for i, name in enumerate(field_names, start=1):
        print(f'  Variable {i} = String "{name}"')
        print(f'  Target Variable {i} = String "<elmer variable for {name}>"')
    print('End\n')
