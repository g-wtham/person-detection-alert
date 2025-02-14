'''
1. Start a loop
2. Get each frame -> process each frame using yolov8 model -> check for person detection -> if detected-> bounding boxes -> show frame -> beep sound & person is detected -> 'exit' key
'''

import cv2
from ultralytics import YOLO
import winsound
import pyttsx3
import time
import threading

model = YOLO("./data/yolo11n.pt")
cap = cv2.VideoCapture("./data/test.mp4")

engine = pyttsx3.init() # TTS only need to be initialized once so it's outside the speak() function

def speak():
    winsound.Beep(2500, 500)
    engine.say("Person is Detected")
    engine.runAndWait()

last_alert_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    person_detected = False
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:
                person_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, "Person", (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 0,0), 2)
                
    current_time = time.time()
    if person_detected and (current_time - last_alert_time > 3):
        threading.Thread(target=speak, daemon=True).start()
        last_alert_time = current_time
        
    cv2.imshow("Person", frame)
    
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()



