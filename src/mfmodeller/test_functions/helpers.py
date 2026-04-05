from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

FloatArray = NDArray[np.float64]
Bounds = list[tuple[float, float]]
MeshCoordinates = tuple[FloatArray, ...]


def x_test_grid_gen(
    ndim: int,
    bounds: Bounds,
    x_test_size: int,
    seed: int = 0,
) -> FloatArray:
    """Create a sorted Latin Hypercube test grid."""
    lower_bounds = [lower for lower, _ in bounds]
    upper_bounds = [upper for _, upper in bounds]

    grid = qmc.LatinHypercube(d=ndim, seed=seed).random(n=x_test_size)
    grid = np.asarray(qmc.scale(grid, lower_bounds, upper_bounds), dtype=float)

    sort_keys = [grid[:, index] for index in reversed(range(grid.shape[1]))]
    sorted_indices = np.lexsort(sort_keys)
    return grid[sorted_indices]


def x_test_mesh_gen(
    ndim: int,
    bounds: Bounds,
    n_points_per_dim: int,
) -> MeshCoordinates:
    """Create a mesh grid for contour plotting."""
    coords_1d: list[FloatArray] = []

    if ndim == 3:
        for index in range(2):
            lower, upper = bounds[index]
            coords_1d.append(
                np.linspace(lower, upper, n_points_per_dim, dtype=float)
            )

        mesh_x, mesh_y = np.meshgrid(coords_1d[0], coords_1d[1], indexing="ij")
        lower, upper = bounds[2]
        avg_value = (lower + upper) / 2
        mesh_z = np.full_like(mesh_x, avg_value)
        return mesh_x, mesh_y, mesh_z

    for lower, upper in bounds[:ndim]:
        coords_1d.append(
            np.linspace(lower, upper, n_points_per_dim, dtype=float)
        )

    mesh_coords = np.meshgrid(*coords_1d, indexing="ij")
    return tuple(np.asarray(coord, dtype=float) for coord in mesh_coords)


def x_test_mesh_flatten(mesh_coords: MeshCoordinates) -> FloatArray:
    """Flatten mesh coordinates into point rows."""
    flattened_coords = [coord.ravel() for coord in mesh_coords]
    return np.column_stack(flattened_coords)


def x_test_mesh_reshape(
    values: FloatArray,
    mesh_shape: tuple[int, ...],
) -> FloatArray:
    """Reshape flattened values back to a mesh grid."""
    return values.reshape(mesh_shape)
