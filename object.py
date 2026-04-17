import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions
from mediapipe.tasks.python.vision import RunningMode, Image

# Initialize object detector options
options = ObjectDetectorOptions(
    base_options=vision.BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
    score_threshold=0.5,
    running_mode=RunningMode.VIDEO
)

# Create the object detector
detector = ObjectDetector.create_from_options(options)

# Start video capture
cap = cv2.VideoCapture(0)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=Image.ImageFormat.SRGB, data=frame_rgb)

    # Run object detection
    detection_result = detector.detect_for_video(mp_image, frame_idx)
    frame_idx += 1

    # Draw detections
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        category = detection.categories[0].category_name
        score = detection.categories[0].score

        # Draw rectangle and label
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{category}: {int(score * 100)}%"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("Object Detection (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
