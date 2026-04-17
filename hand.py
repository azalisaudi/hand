import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)



def detect_peace_sign(hand_landmarks):
    """
    Detects if the hand gesture is a peace sign.
    Returns True if peace sign is detected.
    """

    # Landmark IDs for fingertips and PIP joints
    tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

    # Helper to check if finger is up
    def is_finger_up(tip_id):
        return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y

    # Detect finger states
    fingers = []

    # Thumb: left/right instead of up/down
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index to pinky
    for i in range(1, 5):
        fingers.append(1 if is_finger_up(tip_ids[i]) else 0)

    # Peace sign pattern: thumb down, index & middle up, ring & pinky down
    return fingers == [0, 1, 1, 0, 0]

def detect_fist(hand_landmarks):
    """
    Detects if the hand gesture is a fist (all fingers folded).
    Returns True if a fist is detected.
    """

    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    fingers = []

    # Thumb: Check if folded across palm (x-axis check)
    if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(0)
    else:
        fingers.append(1)

    # For the rest of the fingers (index to pinky)
    for i in range(1, 5):
        # Tip below PIP joint means folded
        if hand_landmarks.landmark[tip_ids[i]].y > hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(0)
        else:
            fingers.append(1)

    # Fist: All fingers down
    return fingers == [0, 0, 0, 0, 0]


def detect_palm(hand_landmarks):
    """
    Detects if the hand gesture is an open palm (all fingers extended).
    Returns True if palm is detected.
    """
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    fingers = []

    # Thumb (check x-axis for horizontal extension)
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index to pinky (check y-axis for vertical extension)
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Palm: all fingers up
    return fingers == [1, 1, 1, 1, 1]



def count_fingers_up(hand_landmarks):
    """
    Returns the number of fingers up based on landmark positions.
    """
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (x-axis check)
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers (y-axis check)
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)  # Total fingers up


while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Flip the image for a mirror effect
    image = cv2.flip(image, 1)
    
    # Convert the image to RGB (MediaPipe expects RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmark coordinates
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # Example: Print the coordinates of the tip of the index finger (landmark 8)
                if idx == 8:
                    print(f"Index finger tip at ({cx}, {cy})")

            # Check the y-coordinates of relevant landmarks
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y


            if detect_peace_sign(hand_landmarks):
                cv2.putText(image, "Peace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            if detect_fist(hand_landmarks):
                cv2.putText(image, "Fist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif thumb_tip_y < index_tip_y and thumb_tip_y < middle_tip_y:
                cv2.putText(image, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                
            if detect_palm(hand_landmarks):               
                cv2.putText(image, "Palm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            if count_fingers_up(hand_landmarks) == 1:
                cv2.putText(image, "1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if count_fingers_up(hand_landmarks) == 3:
                cv2.putText(image, "3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif count_fingers_up(hand_landmarks) == 4:
                cv2.putText(image, "4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break
        
        
cap.release()
cv2.destroyAllWindows()

