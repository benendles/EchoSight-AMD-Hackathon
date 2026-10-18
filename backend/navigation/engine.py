"""Conservative local obstacle policy. No cloud dependency or safety guarantees."""

OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "chair", "bench", "dining table", "table", "couch", "bed", "toilet",
    "refrigerator", "suitcase", "backpack", "dog", "cat", "horse", "cow",
    "sheep", "potted plant", "fire hydrant", "stop sign", "parking meter",
}


def get_object_priority(object_info):
    score = float(object_info.get("path_score", 0))
    distance_weight = {"close": .3, "near": .3, "medium": .15}.get(object_info.get("distance"), 0)
    # Do not cap before sorting: urgent obstacles must retain their ordering.
    return score + distance_weight + (.1 if object_info.get("class") in OBSTACLE_CLASSES else 0) + .05 * float(object_info.get("confidence", 0))


def find_primary_obstacle(scene):
    candidates = []
    for obj in scene:
        if float(obj.get("confidence", 0)) < .45 or not obj.get("in_walking_path"):
            continue
        substantial = obj.get("class") in OBSTACLE_CLASSES or (
            float(obj.get("box_area_ratio", 0)) >= .05 and float(obj.get("path_score", 0)) >= .75
        )
        if substantial:
            candidates.append(obj)
    return max(candidates, key=lambda obj: (obj.get("distance") in {"close", "near"}, get_object_priority(obj)), default=None)


def make_navigation_decision(scene, *, depth_available=True):
    obstacle = find_primary_obstacle(scene)
    if obstacle is None:
        if not depth_available:
            return {"action": "CAUTION", "priority": .5,
                    "message": "Caution. Distance estimates are unavailable; I cannot assess the path reliably."}
        return {"action": "CONTINUE", "priority": 0.0,
                "message": "No immediate obstacle detected. This does not confirm the path is safe."}
    distance = obstacle.get("distance", "unknown")
    position = obstacle.get("position", "ahead")
    object_class = obstacle.get("class", "object")
    # Unknown depth cannot establish proximity; no instructions into unseen routes.
    urgent = distance in {"close", "near"} and float(obstacle.get("path_score", 0)) >= .6
    action = "STOP" if urgent else "CAUTION"
    message = f"{'Stop' if urgent else 'Caution'}. A {object_class} is {position}."
    if distance == "unknown":
        message += " Its distance is uncertain."
    return {"action": action, "priority": min(1., get_object_priority(obstacle)),
            "object": object_class, "position": position, "distance": distance, "message": message}


def describe_navigation(decision):
    return decision["message"]
