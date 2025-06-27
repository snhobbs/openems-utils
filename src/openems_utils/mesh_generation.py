from typing import Sequence, Tuple, List


def _generate_axis_mesh(
    lims: Sequence[float],
    fine_ranges: Sequence[Tuple[float, float]],
    fine_step: float,
    coarse_mult: int
) -> List[float]:
    """
    Generate a 1D mesh across `lims`, using fine spacing in `fine_ranges` and coarse elsewhere.

    Args:
        lims: Start and stop limits [min, max].
        fine_ranges: List of (start, stop) tuples where fine spacing is used.
        coarse_mult: Multiplier for coarse spacing relative to fine_step.
        fine_step: Spacing in fine range.

    Returns:
        Sorted list of mesh points covering lims.
    """
    if coarse_mult < 1:
        raise ValueError(f"coarse_mult must be >= 1, got {coarse_mult}")

    coarse_step = fine_step * coarse_mult
    start, stop = sorted(lims)
    points = [start]
    v = start

    def in_any_fine_range(val: float) -> bool:
        return any(r_start <= val <= r_end for r_start, r_end in fine_ranges)

    while v < stop:
        step = fine_step if in_any_fine_range(v) else coarse_step
        v_next = v + step
        # If we step into a fine range, align to fine edge
        for r_start, _ in fine_ranges:
            if v < r_start <= v_next:
                v_next = r_start
                break
        v = round(v_next, 12)
        if v <= stop:
            points.append(v)

    return sorted(set(points))


def generate_symmetric_axis_mesh(
    lims: Sequence[float],
    fine_range: Tuple[float, float],
    fine_step: float,
    coarse_mult: int
) -> List[float]:
    """
    Generate a symmetric mesh centered at 0 for one axis.
    Useful for placing a port about 0 and having symmetrical lines coming out of them.

    Args:
        lims: Full range of the axis (e.g., xlims), should include negative and positive bounds.
        fine_range: Tuple (start, stop) for fine resolution region (centered around 0).
        fine_step: Step size in fine region.
        coarse_mult: Coarse step multiplier relative to fine_step.

    Returns:
        Sorted list of mesh points spanning both positive and negative sides.
    """
    fine_start, fine_stop = sorted(abs(x) for x in fine_range)
    pos_mesh = _generate_axis_mesh([0, max(lims)], [(fine_start, fine_stop)],
                                    fine_step=fine_step, coarse_mult=coarse_mult)
    neg_mesh = [-v for v in pos_mesh if v != 0]
    return sorted(neg_mesh + pos_mesh)


def generate_cartesian_meshes(
    lims: Sequence[Sequence[float]],
    port_start: Sequence[float],
    port_stop: Sequence[float],
    fine_step: float,
    coarse_mult: int
) -> Tuple[List[float], List[float], List[float]]:
    """
    Generate symmetric X, Y, Z mesh grids with finer resolution near defined port regions.

    Usage:
    Assuming a single input port that needs fine resolution
    ```python
    # symmetrical -> 0 -> xlims and double it in the negative
    meshx, meshy, meshz = generate_cartesian_meshes(
                    [xlims, ylims, zlims],
                    port_start=port1_start,
                    port_stop=port1_stop,
                    fine_step=fine_grid,
                    coarse_mult=course_grid_mult
                )
    openEMS_grid.AddLine('x', meshx)
    openEMS_grid.AddLine('y', meshy)
    openEMS_grid.AddLine('z', meshz)
    ```

    Args:
        lims: Sequence of (min, max) tuples for X, Y, Z axes.
        port_start: (x, y, z) starting coordinates for port region.
        port_stop: (x, y, z) ending coordinates for port region.
        fine_step: Step size in fine region.
        coarse_mult: Coarse step multiplier relative to fine_step.

    Returns:
        Tuple of mesh lists: (mesh_x, mesh_y, mesh_z)
    """
    if len(lims) != 3 or len(port_start) != 3 or len(port_stop) != 3:
        raise ValueError("lims, port_start, and port_stop must all be length-3 sequences.")

    meshes = []
    for i in range(3):
        fine_extent = max(abs(port_start[i]), abs(port_stop[i])) + fine_step
        mesh = generate_symmetric_axis_mesh(
            lims=lims[i],
            fine_range=(0, fine_extent),
            fine_step=fine_step,
            coarse_mult=coarse_mult
        )
        meshes.append(mesh)

    return tuple(sorted(meshes))  # (mesh_x, mesh_y, mesh_z)
