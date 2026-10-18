from vision.detector import detect
from vision.depth import estimate_depth
from vision.scene import (
    analyze_scene,
    describe_scene
)
from navigation.engine import (
    make_navigation_decision,
    describe_navigation
)


IMAGE_PATH = "bus.jpg"


print("\n==============================")
print("       ECHOSIGHT TEST")
print("==============================\n")


# 1. OBJECT DETECTION

print("Running object detection...")

detections = detect(IMAGE_PATH)

print(f"Detected {len(detections)} objects.\n")


for detection in detections:

    print(
        f"Object: {detection['class']}"
    )

    print(
        f"Confidence: "
        f"{detection['confidence']:.2f}"
    )

    print(
        f"Direction: "
        f"{detection['direction']}"
    )

    print(
        f"Vertical: "
        f"{detection['vertical_position']}"
    )

    print(
        f"Box: "
        f"{detection['box']}"
    )

    print()


# 2. DEPTH ESTIMATION

print("Running depth estimation...\n")

depth_map = estimate_depth(
    IMAGE_PATH
)

print(
    f"Depth map shape: "
    f"{depth_map.shape}"
)

print(
    f"Depth minimum: "
    f"{depth_map.min():.4f}"
)

print(
    f"Depth maximum: "
    f"{depth_map.max():.4f}"
)

print()


# 3. SPATIAL ENGINE

print("Running spatial engine...\n")

scene = analyze_scene(
    detections,
    depth_map
)



# 4. STRUCTURED SCENE

print("==============================")
print("       SCENE STATE")
print("==============================\n")

for obj in scene:

    print(
        f"Object: {obj['class']}"
    )

    print(
        f"Confidence: "
        f"{obj['confidence']:.2f}"
    )

    print(
        f"Direction: "
        f"{obj['direction']}"
    )

    print(
        f"Vertical: "
        f"{obj['vertical_position']}"
    )

    print(
        f"Relative depth: "
        f"{obj['relative_depth']}"
    )

    # --------------------------------
    # Estimated metric distance
    # --------------------------------

    if obj["distance_meters"] is not None:

        print(
            f"Distance: "
            f"{obj['distance_meters']:.2f} meters"
        )

    else:

        print(
            "Distance: unknown"
        )

    # --------------------------------
    # Navigation reasoning
    # --------------------------------

    print(
        f"Path score: "
        f"{obj['path_score']:.2f}"
    )

    print(
        f"Box area ratio: "
        f"{obj['box_area_ratio']:.2%}"
    )

    print(
        f"In walking path: "
        f"{obj['in_walking_path']}"
    )

    print(
        f"Box: "
        f"{obj['box']}"
    )

    print("------------------------------")

# 5. NAVIGATION ENGINE

print("\n==============================")
print("       NAVIGATION")
print("==============================\n")

decision = make_navigation_decision(
    scene
)

print(
    f"Action: "
    f"{decision['action']}"
)

print(
    f"Priority: "
    f"{decision['priority']:.2f}"
)

if "object" in decision:

    print(
        f"Object: "
        f"{decision['object']}"
    )

    print(
        f"Position: "
        f"{decision['position']}"
    )

    print(
        f"Distance: "
        f"{decision['distance']}"
    )

print(
    f"Message: "
    f"{describe_navigation(decision)}"
)