import numpy as np
import copy


def make_1d_ob_grid(start, stop, step, OB):
    vol = np.log10(np.cumsum(OB[:, 1]))

    bins = np.arange(start, stop + step, step)
    inds = np.digitize(vol, bins)

    grid = np.zeros(len(bins))
    # the same index can occur multiple times in 'inds', so,
    # reversing the arrays means that the smallest rate will be used,
    # instead of the largest
    grid[np.flip(inds)] = np.flip(OB[:, 0])

    return grid


def get_grid_ob(start, stop, step, *, mid_rate, OB):
    # """"""

    OB = copy.copy(OB)

    def monotonic(x):
        dx = np.diff(x)
        return np.all(dx <= 0) or np.all(dx >= 0)

    assert monotonic(OB[:, 0]), "OB rates must be monotonic"

    # Convert quantity to base currency
    OB[:, 1] = OB[:, 0] * OB[:, 1]

    # Get the rate as a ratio of mp
    OB[:, 0] = np.abs(1 - (OB[:, 0] / mid_rate))

    # represent OB as 1D grid on the Preceding Volume axis
    OB = make_1d_ob_grid(start, stop, step, OB)

    if OB[0] == 0:
        OB[0] = 1e-8  # fixes log(0) problem
    OB = np.maximum.accumulate(OB)
    OB = np.log10(OB)

    assert monotonic(OB), "OB rates must be monotonic"

    return OB
