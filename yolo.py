import cv2
from ultralytics import YOLO

# Load YOLOv5 or YOLOv8 model (pretrained on COCO)
model = YOLO("yolov5s.pt")  # or "yolov8n.pt"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

