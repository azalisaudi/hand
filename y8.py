from ultralytics import YOLO
import cv2

yolo_ver = input("Which yolo: ")
if yolo_ver == "3":
    yolo_ver = "yolov3-tiny.pt"
elif yolo_ver == "5":
    yolo_ver = "yolov5n.pt"
elif yolo_ver == "8":
    yolo_ver = "yolov8n.pt"
elif yolo_ver == "12":
    yolo_ver = "yolo12n.pt"

# Load a pretrained YOLOv8 model
model = YOLO(yolo_ver)  # Use 'yolov8s.pt', 'yolov8m.pt', etc., for larger models

camera_index = input("Which camera: ")
camera_index = int(camera_index)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection on the current frame
    results = model.predict(source=frame, show=False, conf=0.3, verbose=False)

    # Draw detections on the frame
    annotated_frame = results[0].plot()

    # Show the annotated frame
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
