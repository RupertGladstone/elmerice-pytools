# elmerice-pytools

Shared Python helpers for working with Elmer/Ice VTU output and meshes --
regridding, mesh I/O, node-sampling.

## Layout

- `elmerice_pytools/vtu_mesh.py` -- loading, welding, surface extraction
  (`GeometryIds`), and point-sampling from Elmer VTU/PVTU output.
- `elmerice_pytools/elmer_mesh.py` -- reading/writing Elmer's plain-text
  mesh format, boundary-edge topology, mesh fingerprinting.
- `elmerice_pytools/regrid.py` -- regridding onto a structured x/y grid
  (for `GridDataReader`).
- `elmerice_pytools/netcdf_io.py` -- writing the two netCDF shapes this
  project family reads back into Elmer: node-matching (`UGridDataReader`)
  or structured-grid (`GridDataReader`).
- `elmerice_pytools/cli/` -- command-line tools built on the above:
  - `pvtu_to_netcdf.py` -- regrid a VTU/PVTU onto a structured grid.
  - `vtu_to_node_netcdf.py` -- sample a VTU/PVTU's fields directly at
    another (dimensionally different) mesh's own node positions, e.g. for
    seeding a 2D SSA run's state from a 3D BP/FS run's output, where
    Elmer's own `Restart File` can't bridge the two meshes.

## Install

```
pip install -e /home/rupert/Source/elmerice-pytools
```
