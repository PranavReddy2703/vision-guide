import cv2
import pyttsx3
import time
import threading
import queue
from ultralytics import YOLO

# 1. Initialization
print("Loading YOLO model...")
model = YOLO("yolov8l.pt")
audio_queue = queue.Queue()

def tts_worker():
    try:
        engine = pyttsx3.init(driverName='sapi5')
    except:
        engine = pyttsx3.init()
    
    engine.setProperty('rate', 160)
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)

    while True:
        item = audio_queue.get()
        if item is None:
            break
        
        text, urgent = item
        print(f"[TTS] Speaking: {text}")
        try:
            engine.stop()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")
        audio_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text, urgent=False):
    if urgent:
        with audio_queue.mutex:
            audio_queue.queue.clear()
    if text and text.strip():
        audio_queue.put((text, urgent))

# 2. Setup
print("Initializing camera with 16:9 aspect ratio...")
cap = cv2.VideoCapture(0)

# Requesting a 16:9 resolution (1080p)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Camera access failed.")
    exit()

actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to: {actual_w}x{actual_h}")

# Ensure the window handles the aspect ratio correctly without squeezing
cv2.namedWindow("Visual Assistant", cv2.WINDOW_AUTOSIZE)
cv2.setWindowProperty("Visual Assistant", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

speak("System ready. Starting real-time hazard tracking.")
print("System Ready. Press 'q' in the window to quit.")

last_spoken_time = 0
cooldown_seconds = 5
track_history = {}
last_urgent_warnings = {} 

HAZARD_CLASSES = {'car', 'motorcycle', 'bus', 'truck', 'bicycle'}
GROWTH_THRESHOLD = 1.20
URGENT_COOLDOWN = 3.0

# 3. Processing Loop
try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Warning: Failed to read frame from camera.")
            time.sleep(0.1)
            continue


        results = model.track(frame, persist=True, verbose=False)
        detected_objects = set()
        current_time = time.time()

        if results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().tolist()
            boxes_xywh = results[0].boxes.xywh.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()

            for coords, dims, tid, cls_idx in zip(boxes_xyxy, boxes_xywh, track_ids, class_indices):
                class_name = model.names[cls_idx]
                detected_objects.add(class_name)

                x1, y1, x2, y2 = map(int, coords)
                area = int(dims[2] * dims[3])

                print(f"[Coords] ID {tid} ({class_name}): TL({x1},{y1}), TR({x2},{y1}), BL({x1},{y2}), BR({x2},{y2}) | Area: {area}px")

                if tid not in track_history:
                    print(f"[New Track] ID: {tid} | Class: {class_name}")
                    track_history[tid] = []
                
                history = track_history[tid]
                history.append((current_time, area))
                history = [h for h in history if current_time - h[0] <= 0.6]
                track_history[tid] = history

                is_approaching = False
                if class_name in HAZARD_CLASSES and len(history) >= 3:
                    old_time, old_area = history[0]
                    if (current_time - old_time) >= 0.2 and old_area > 0:
                        if (area / old_area) > GROWTH_THRESHOLD:
                            is_approaching = True
                            if tid not in last_urgent_warnings or (current_time - last_urgent_warnings[tid] > URGENT_COOLDOWN):
                                print(f"[HAZARD] {class_name} (ID: {tid}) is approaching fast!")
                                speak(f"Urgent! {class_name} approaching fast!", urgent=True)
                                last_urgent_warnings[tid] = current_time

                color = (0, 0, 255) if is_approaching else (0, 255, 0)
                status = "FAST " if is_approaching else ""
                
                tl = f"({x1},{y1})"
                tr = f"({x2},{y1})"
                bl = f"({x1},{y2})"
                br = f"({x2},{y2})"
                
                main_label = f"{status}{class_name} | {area}px"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, main_label, (x1, y1 - 25), font, 0.5, color, 2)
                cv2.putText(frame, tl, (x1, y1 - 5), font, 0.4, color, 1)
                cv2.putText(frame, tr, (x2 - 40, y1 - 5), font, 0.4, color, 1)
                cv2.putText(frame, bl, (x1, y2 + 15), font, 0.4, color, 1)
                cv2.putText(frame, br, (x2 - 40, y2 + 15), font, 0.4, color, 1)

        dead_ids = [tid for tid, hist in track_history.items() if current_time - hist[-1][0] > 2.0]
        for tid in dead_ids:
            print(f"[Lost Track] ID: {tid}")
            track_history.pop(tid, None)
            last_urgent_warnings.pop(tid, None)

        if detected_objects and (current_time - last_spoken_time > cooldown_seconds):
            print(f"[Env Update] {', '.join(detected_objects)}")
            speak(f"Environment: {', '.join(detected_objects)}")
            last_spoken_time = current_time

        display_frame = cv2.resize(frame, (1920, 1080)) 
        cv2.imshow("Visual Assistant", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Visual Assistant", cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("\nManual stop triggered.")

# 4. Shutdown
print("Shutting down system...")
speak("Shutting down.", urgent=True)
audio_queue.put(None)
cap.release()
cv2.destroyAllWindows()
print("Resources released. Goodbye.")