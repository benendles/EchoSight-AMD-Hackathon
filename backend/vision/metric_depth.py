import numpy as np
import os
# Calibration

# These are REFERENCE measurements.

# Example:
# relative depth  -> actual distance

# Replace these values with measurements from
# your own camera/images.

# IMPORTANT:
# Depth Anything V2's depth scale is not guaranteed
# to be linear in meters, so we fit a calibration model.
CALIBRATION_POINTS = [
    # (relative_depth, distance_in_meters)
    (2.93, 5.0),
    (3.50, 4.0),
    (4.20, 3.0),
    (4.80, 2.0),
]


def calibrate_depth(relative_depth):
    """
    Convert relative Depth Anything V2 output
    into an estimated distance in meters.

    Returns:
        float: estimated distance in meters
    """

    # Sample calibration points are not camera measurements. Explicitly opt in
    # only after replacing CALIBRATION_POINTS with measured camera calibration.
    if os.getenv("ECHOSIGHT_METRIC_DEPTH_ENABLED", "false").lower() != "true":
        return None
    if relative_depth is None or not np.isfinite(relative_depth):
        return None

    depths = np.array(
        [point[0] for point in CALIBRATION_POINTS],
        dtype=np.float32
    )

    distances = np.array(
        [point[1] for point in CALIBRATION_POINTS],
        dtype=np.float32
    )

    # Fit a polynomial relationship.
    #
    # Degree 1 = linear calibration.
    #
    coefficients = np.polyfit(
        depths,
        distances,
        1
    )

    distance = np.polyval(
        coefficients,
        relative_depth
    )

    # Distance cannot be negative.
    distance = max(0.0, float(distance))

    return distance
