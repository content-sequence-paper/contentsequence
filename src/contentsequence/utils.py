import pytensor.tensor as pt
import numpy as np

def _global_extrema_locations(candidate_locations, candidate_positions, candidate_valid):
    """Return earliest global minimum and maximum locations from valid candidates."""
    valid_minimum_positions = pt.where(
        candidate_valid,
        candidate_positions,
        np.inf,
    )
    minimum_position = pt.min(valid_minimum_positions)
    minimum_location = pt.min(
        pt.where(
            pt.and_(
                candidate_valid,
                pt.eq(candidate_positions, minimum_position),
            ),
            candidate_locations,
            np.inf,
        )
    )

    valid_maximum_positions = pt.where(
        candidate_valid,
        candidate_positions,
        -np.inf,
    )
    maximum_position = pt.max(valid_maximum_positions)
    maximum_location = pt.min(
        pt.where(
            pt.and_(
                candidate_valid,
                pt.eq(candidate_positions, maximum_position),
            ),
            candidate_locations,
            np.inf,
        )
    )
    return minimum_location, maximum_location
