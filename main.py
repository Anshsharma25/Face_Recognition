# import cv2
# import numpy as np
# import face_recognition
# import os
# from datetime import datetime

# # from PIL import ImageGrab

# path = 'Training_image'
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)
# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)


# def findEncodings(images):
#     encodeList = []


#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList


# def markAttendance(name):
#     with open('Attendance.csv', 'r+') as f:
#         myDataList = f.readlines()


#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#             if name not in nameList:
#                 now = datetime.now()
#                 dtString = now.strftime('%H:%M:%S')
#                 f.writelines(f'\n{name},{dtString}')

# #### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# # def captureScreen(bbox=(300,300,690+300,530+300)):
# #     capScr = np.array(ImageGrab.grab(bbox))
# #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
# #     return capScr

# encodeListKnown = findEncodings(images)
# print('Encoding Complete')

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
# # img = captureScreen()
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# # print(faceDis)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
# # print(name)
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#             markAttendance(name)

#     cv2.imshow('Webcam', img)
#     cv2.waitKey(1)

'''
====================================================================================================
Here's the code start For the multiple picture of user (Front, Right, Left , Down)
====================================================================================================

'''

import cv2
import cv2
import os
import face_recognition
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────
KNOWN_DIR = "Training_image"
TOLERANCE = 0.5  # Lower is stricter
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"  # or "cnn" for GPU

# ── HELPER FUNCTIONS ─────────────────────────────────
def load_known_faces():
    known_faces = []
    known_names = []

    for name in os.listdir(KNOWN_DIR):
        person_dir = os.path.join(KNOWN_DIR, name)
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_faces.append(encoding[0])
                known_names.append(name)
    return known_faces, known_names

def get_angle_label(face_landmarks):
    try:
        nose = face_landmarks['nose_tip'][0]
        left_eye = face_landmarks['left_eye'][0]
        right_eye = face_landmarks['right_eye'][0]

        eye_diff = right_eye[0] - left_eye[0]
        nose_to_eye_center = abs(nose[0] - (left_eye[0] + right_eye[0]) / 2)

        if nose_to_eye_center > 20:
            return "Side (Left)" if nose[0] > right_eye[0] else "Side (Right)"
        else:
            return "Front"
    except:
        return "Unknown"

# ── MAIN ─────────────────────────────────────────────
known_faces, known_names = load_known_faces()
true_positives = 0
false_positives = 0

cap = cv2.VideoCapture(0)

print("Starting webcam. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model=MODEL)
    encodings = face_recognition.face_encodings(rgb, locations)
    landmarks = face_recognition.face_landmarks(rgb)

    for encoding, location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, encoding, TOLERANCE)
        distances = face_recognition.face_distance(known_faces, encoding)

        match_index = None
        best_confidence = 0

        if True in results:
            match_index = results.index(True)
            name = known_names[match_index]
            confidence = round((1 - distances[match_index]) * 100, 2)
            best_confidence = confidence

            true_positives += 1
            result_text = f"{name} ({confidence}%)"
            print(f"[TRUE POSITIVE] Matched {name} | Confidence: {confidence}%")

        else:
            false_positives += 1
            result_text = "Unknown"
            print(f"[FALSE POSITIVE] No match found")

        # Draw rectangle
        top_left = (location[3], location[0])
        bottom_right = (location[1], location[2])
        color = (0, 255, 0) if match_index else (0, 0, 255)
        cv2.rectangle(frame, top_left, bottom_right, color, FRAME_THICKNESS)

        # Draw label
        cv2.putText(frame, result_text, (location[3], location[2] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, FONT_THICKNESS)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n--- Summary ---")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")


