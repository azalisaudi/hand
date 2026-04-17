import cv2
import numpy as np
import time
from ffpyplayer.player import MediaPlayer
import mediapipe as mp

# --- Load video and detect FPS ---
video_path = "transformers.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
if fps <= 0 or fps > 240:
    fps = 30
frame_interval = 1.0 / fps

# --- MediaPipe Hands setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# --- Webcam for gesture detection ---
webcam = cv2.VideoCapture(0)

def is_open_palm(landmarks):
    """3 or more fingers open (excluding thumb)"""
    tip_ids = [8, 12, 16, 20]
    open_fingers = 0
    for tip_id in tip_ids:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            open_fingers += 1
    return open_fingers >= 3

def is_thumbs_up(landmarks):
    """Detect thumbs-up gesture"""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    fingers_curled = True

    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers_curled = False
            break

    thumb_up = thumb_tip.y < thumb_ip.y
    return fingers_curled and thumb_up

def is_peace_sign(landmarks):
    """Index and middle finger extended; rest curled"""
    extended = 0
    curled = 0
    for i, tip_id in enumerate([8, 12, 16, 20]):
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            if tip_id in [8, 12]:
                extended += 1
            else:
                curled += 1
        else:
            if tip_id in [16, 20]:
                curled += 1
    return extended == 2 and curled == 2

def detect_gesture(frame):
    """Return 'pause', 'play', or None based on hand gesture"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            if is_open_palm(landmarks):
                return 'pause'
            elif is_thumbs_up(landmarks) or is_peace_sign(landmarks):
                return 'play'
    return None

def get_frame(player):
    frame, val = player.get_frame()
    if val == 'eof':
        return None
    if frame is None:
        return False
    img, _ = frame
    w, h = img.get_size()
    buf = img.to_bytearray()[0]
    frame_rgb = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

# --- Video player setup ---
player = MediaPlayer(video_path)
paused = False
target_width, target_height = 1280, 720
last_frame_time = time.time()

print("✋ Open hand = pause\n👍 Thumbs-up or ✌️ Peace = play\nPress Q to quit.")

while True:
    ret, cam_frame = webcam.read()
    if not ret:
        break

    gesture = detect_gesture(cam_frame)

    if gesture == 'pause' and not paused:
        paused = True
        player.set_pause(paused)
        last_frame_time = time.time()
        print("Paused")

    elif gesture == 'play' and paused:
        paused = False
        player.set_pause(paused)
        last_frame_time = time.time()
        print("Playing")

    if not paused:
        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        frame = get_frame(player)
        last_frame_time = time.time()

        if frame is None:
            break
        elif frame is False:
            continue

        frame = cv2.resize(frame, (target_width, target_height))
        #cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Gesture-Controlled Player", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
player.close_player()
hands.close()
