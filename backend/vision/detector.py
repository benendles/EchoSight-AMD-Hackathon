from ultralytics import YOLO
from pathlib import Path

# Load YOLO11n
MODEL_PATH = Path(__file__).resolve().parents[2] / "yolo11n.pt"
model = None


def detect(image_path):
    """
    Detect objects in an image using YOLO11n.
    """

    global model
    if model is None:
        model = YOLO(str(MODEL_PATH))
    results = model(image_path, verbose=False)
    result = results[0]

    width = result.orig_shape[1]
    height = result.orig_shape[0]

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = result.names[class_id]

            # Bounding box center

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Horizontal position

        if center_x < width / 3:
            direction = "left"

        elif center_x < 2 * width / 3:
            direction = "center"

        else:
            direction = "right"

        # Vertical position

        if center_y < height / 3:
            vertical_position = "above"

        elif center_y < 2 * height / 3:
            vertical_position = "center"

        else:
            vertical_position = "below"

        # Detection

        detection = {
            "class": class_name,
            "confidence": confidence,
            "box": [x1, y1, x2, y2],
            "direction": direction,
            "vertical_position": vertical_position
        }

        detections.append(detection)

    return detections
