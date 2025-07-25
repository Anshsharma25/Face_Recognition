import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'Training_image'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)


# import cv2
# import numpy as np
# import face_recognition
# import os
# import requests
# from datetime import datetime

# # ─── CONFIG ──────────────────────────────────────────────────────
# TRAIN_DIR      = 'Training_image'             # your folder of known faces
# USE_IP_CAMERA  = True                         # False → use local webcam
# IP_SNAPSHOT_URL = 'http://192.168.1.104/cam-hi.jpg'
# CSV_FILE       = 'Attendance.csv'
# CSV_HEADER     = 'Name,Time\n'

# # ─── HELPERS ─────────────────────────────────────────────────────
# def ensure_csv():
#     if not os.path.exists(CSV_FILE):
#         with open(CSV_FILE, 'w', newline='') as f:
#             f.write(CSV_HEADER)

# def mark_attendance(name):
#     ensure_csv()
#     with open(CSV_FILE, 'r+', newline='') as f:
#         lines = f.readlines()
#         logged = { line.split(',')[0] for line in lines }
#         if name not in logged:
#             now = datetime.now().strftime('%H:%M:%S')
#             f.write(f'{name},{now}\n')

# def load_known_faces(folder):
#     encs, names = [], []
#     for fname in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder, fname))
#         if img is None:
#             print(f"⚠️ Could not read {fname}, skipping.")
#             continue
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         locs = face_recognition.face_locations(rgb)
#         if not locs:
#             print(f"⚠️ No face in {fname}, skipping.")
#             continue
#         enc = face_recognition.face_encodings(rgb, locs)[0]
#         encs.append(enc)
#         names.append(os.path.splitext(fname)[0])
#     return encs, names

# # ─── INIT ────────────────────────────────────────────────────────
# known_encs, known_names = load_known_faces(TRAIN_DIR)
# print(f"✅ Loaded {len(known_encs)} known faces: {known_names}")
# if not known_encs:
#     raise SystemExit("No known faces found – check your Training_image folder.")

# # If using webcam:
# if not USE_IP_CAMERA:
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise SystemExit("Could not open local webcam.")

# print("\n❗ Press 'c' to capture & recognize a single frame, 'q' to quit.\n")

# # ─── MAIN LOOP ───────────────────────────────────────────────────
# while True:
#     if USE_IP_CAMERA:
#         # fetch snapshot on demand
#         # we’ll only fetch when user presses 'c'
#         frame = None
#         display = np.zeros((200,400,3), dtype=np.uint8)
#         cv2.putText(display, "Press 'c' to capture IP snapshot", (10,100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
#         cv2.imshow('FaceRec Capture', display)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('c'):
#             try:
#                 resp = requests.get(IP_SNAPSHOT_URL, timeout=5)
#                 img_arr = np.frombuffer(resp.content, np.uint8)
#                 frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
#             except Exception as e:
#                 print(f"⚠️ Snapshot error: {e}")
#                 frame = None
#         elif key == ord('q'):
#             break
#     else:
#         # webcam mode: continuously show feed, capture on 'c'
#         ret, frame = cap.read()
#         cv2.imshow('FaceRec Capture', frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('c'):
#             pass  # frame already set
#         elif key == ord('q'):
#             break
#         else:
#             continue

#     # if we got a frame on 'c', run recognition
#     if frame is not None:
#         # --- preprocess & detect ---
#         small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
#         rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
#         locs = face_recognition.face_locations(rgb_small)
#         encs = face_recognition.face_encodings(rgb_small, locs)

#         # --- annotate & log ---
#         for enc, loc in zip(encs, locs):
#             dists   = face_recognition.face_distance(known_encs, enc)
#             idx     = np.argmin(dists)
#             matches = face_recognition.compare_faces(known_encs, enc)
#             if matches[idx]:
#                 name = known_names[idx].upper()
#                 y1,x2,y2,x1 = [c*4 for c in loc]
#                 cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
#                 cv2.putText(frame, name, (x1, y1-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#                 mark_attendance(name)
#                 print(f"✅ Recognized: {name}")

#         # show the captured & annotated image
#         cv2.imshow('Captured Image', frame)
#         cv2.waitKey(0)  # wait until any key pressed
#         cv2.destroyWindow('Captured Image')

# # ─── CLEANUP ─────────────────────────────────────────────────────
# if not USE_IP_CAMERA:
#     cap.release()
# cv2.destroyAllWindows()
