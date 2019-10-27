from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
#import playsound
import argparse
import imutils
import time
import dlib
import socket
import sys
import cv2
import pickle
import numpy as np
import struct 
path='cat1.mp3'
HOST='192.168.137.1'
PORT=8089
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print ('Socket created')
s.bind((HOST,PORT))
print ('Socket bind complete')
s.listen(10)
print ('Socket now listening')
conn,addr=s.accept()
data = b''
payload_size = struct.calcsize("L")

def sound_alarm(path):
	# play an alarm sound
	#playsound.playsound(path)
	print("Drowsy driver")
  
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 6
COUNTER = 0
ALARM_ON = False
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
while True:
  while len(data) < payload_size:
    data += conn.recv(4096)
  packed_msg_size = data[:payload_size]
  data = data[payload_size:]
  msg_size = struct.unpack("L", packed_msg_size)[0]
  while len(data) < msg_size:
    data += conn.recv(4096)
  frame_data = data[:msg_size]
  data = data[msg_size:]
  frame=pickle.loads(frame_data)
  #frame = vs.read()
  frame = imutils.resize(frame, width=450)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  rects = detector(gray, 0)
  for rect in rects:
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    if ear < EYE_AR_THRESH:
      COUNTER += 1
      if COUNTER >= EYE_AR_CONSEC_FRAMES:
        if not ALARM_ON:
          ALARM_ON = True
          if path!= "":
            t = Thread(target=sound_alarm,
                       args=(path,))
            t.deamon = True
            t.start()
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
      COUNTER = 0
      ALARM_ON = False
    # draw the computed eye aspect ratio on the frame to help
    # with debugging and setting the correct eye aspect ratio
    # thresholds and frame counters
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

  # show the frame
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
      break


