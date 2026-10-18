import numpy as np


# HORIZONTAL POSITION

def estimate_direction(box, image_width):
    """
    Determine horizontal position of an object.

    Returns:
        left
        center
        right
    """

    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2

    if center_x < image_width / 3:
        return "left"

    elif center_x < 2 * image_width / 3:
        return "center"

    else:
        return "right"


# VERTICAL POSITION

def estimate_vertical_position(box, image_height):
    """
    Determine vertical position of an object.

    Returns:
        above
        center
        below
    """

    x1, y1, x2, y2 = box

    center_y = (y1 + y2) / 2

    if center_y < image_height / 3:
        return "above"

    elif center_y < 2 * image_height / 3:
        return "center"

    else:
        return "below"


# OBJECT SIZE

def calculate_box_area(box):
    """
    Calculate the area of an object's
    bounding box.
    """

    x1, y1, x2, y2 = box

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def calculate_box_area_ratio(
    box,
    image_width,
    image_height
):
    """
    Calculate how much of the image
    is occupied by an object.

    Returns:
        value between 0 and 1
    """

    image_area = image_width * image_height

    if image_area <= 0:
        return 0.0

    box_area = calculate_box_area(box)

    return float(box_area / image_area)


# DEPTH

def get_object_depth(depth_map, box):
    """
    Extract the median relative depth value
    inside an object's bounding box.

    Depth Anything V2 produces relative depth,
    not true metric distance.
    """

    x1, y1, x2, y2 = map(int, box)

    height, width = depth_map.shape

    # Keep coordinates inside image
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))

    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    object_depth = depth_map[y1:y2, x1:x2]

    if object_depth.size == 0:
        return None

    # Remove invalid values
    object_depth = object_depth[
        np.isfinite(object_depth)
    ]

    if object_depth.size == 0:
        return None

    return float(np.median(object_depth))


# RELATIVE DISTANCE

def classify_distance(object_depth, scene_depth):
    """
    Classify an object's depth relative to the scene.

    Returns:
        close
        medium
        far
        unknown
    """

    if object_depth is None:
        return "unknown"

    if scene_depth is None or scene_depth.size == 0 or not np.isfinite(object_depth):
        return "unknown"

    scene_depth = scene_depth[np.isfinite(scene_depth)]
    if scene_depth.size == 0 or np.ptp(scene_depth) < 1e-6:
        return "unknown"

    near_threshold = np.percentile(
        scene_depth,
        70
    )

    far_threshold = np.percentile(
        scene_depth,
        30
    )

    if object_depth >= near_threshold:
        return "close"

    elif object_depth <= far_threshold:
        return "far"

    else:
        return "medium"


# WALKING PATH SCORE

def calculate_path_score(
    direction,
    vertical_position,
    distance,
    box_area_ratio
):
    """
    Estimate how likely an object is to
    obstruct the user's walking path.

    Returns:
        score between 0 and 1
    """

    score = 0.0

    # Horizontal position

    if direction == "center":
        score += 0.50

    elif direction in {"left", "right"}:
        score += 0.10

    # Vertical position

    if vertical_position == "below":
        score += 0.25

    elif vertical_position == "center":
        score += 0.20

    # Distance

    if distance == "close":
        score += 0.20

    elif distance == "medium":
        score += 0.10

    # Object size

    if box_area_ratio >= 0.30:
        score += 0.15

    elif box_area_ratio >= 0.15:
        score += 0.10

    elif box_area_ratio >= 0.05:
        score += 0.05

    return min(score, 1.0)


# WALKING PATH

def is_in_walking_path(
    direction,
    vertical_position,
    distance="unknown",
    box_area_ratio=0.0
):
    """
    Determine whether an object is likely
    to be in the user's immediate walking path.

    Returns:
        True / False
    """

    path_score = calculate_path_score(
        direction,
        vertical_position,
        distance,
        box_area_ratio
    )

    return path_score >= 0.60


# SPATIAL RELATIONSHIP

def determine_spatial_relationship(
    direction,
    vertical_position
):
    """
    Convert horizontal and vertical positions
    into a human-readable spatial relationship.
    """

    if direction == "center":

        if vertical_position == "center":
            return "directly ahead"

        return vertical_position

    if vertical_position == "center":
        return f"to the {direction}"

    return (
        f"{vertical_position} "
        f"and to the {direction}"
    )


# BUILD SPATIAL REPRESENTATION

def build_spatial_description(
    object_class,
    direction,
    vertical_position,
    distance,
    box=None,
    image_width=None,
    image_height=None
):
    """
    Build a structured spatial representation.

    Includes:
        position
        distance
        object size
        path score
        walking-path status
    """

    position = determine_spatial_relationship(
        direction,
        vertical_position
    )

    # --------------------------------
    # Object size
    # --------------------------------

    box_area_ratio = 0.0

    if (
        box is not None
        and image_width is not None
        and image_height is not None
    ):
        box_area_ratio = calculate_box_area_ratio(
            box,
            image_width,
            image_height
        )

    # --------------------------------
    # Path score
    # --------------------------------

    path_score = calculate_path_score(
        direction,
        vertical_position,
        distance,
        box_area_ratio
    )

    in_path = path_score >= 0.60

    return {
        "object": object_class,
        "position": position,
        "direction": direction,
        "vertical_position": vertical_position,
        "distance": distance,

        "box_area_ratio": box_area_ratio,

        "path_score": path_score,

        "in_walking_path": in_path
    }
