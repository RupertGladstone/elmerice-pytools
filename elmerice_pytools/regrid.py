"""Regridding a mesh's point-data fields onto a regular structured x/y grid,
for GridDataReader. Built on vtu_mesh.sample_fields_at_points -- a
structured grid is just a particular (N, 2) point list, so this module adds
only axis construction and the reshape back to (ny, nx)."""
import math

import numpy as np

from .vtu_mesh import sample_fields_at_points


def build_axis(vmin, vmax, dx, ncells_default):
    extent = vmax - vmin
    if dx is None:
        n_cells = ncells_default
        dx = extent / n_cells
        pad = 0.0
    else:
        n_cells = max(1, math.ceil(extent / dx))
        pad = n_cells * dx - extent
    v0 = vmin - pad / 2.0
    n_points = n_cells + 1
    values = v0 + dx * np.arange(n_points)
    return values


def regrid_to_structured_grid(mesh, variables, dx=None, dy=None, ncells=100):
    """Returns (x, y, fields) where fields[name] has shape (len(y), len(x)).
    Grid extent is the mesh's own x/y bounding box; see build_axis for how
    --dx/--ncells determine resolution."""
    xmin, xmax, ymin, ymax, _, _ = mesh.bounds
    x = build_axis(xmin, xmax, dx, ncells)
    y = build_axis(ymin, ymax, dy if dy is not None else dx, ncells)

    nx, ny = len(x), len(y)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack([xx.ravel(), yy.ravel()])

    sampled = sample_fields_at_points(mesh, xy, variables)
    fields = {name: values.reshape(ny, nx) for name, values in sampled.items()}
    return x, y, fields
