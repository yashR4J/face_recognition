#! /usr/bin/env python3

import cv2
from random import randrange

# path=r'/mnt/c/Users/yashr/code/face_recognition/'
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('hemsworth.jpg')
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# for coordinate in face_coordinates:
#     (x, y, w, h) = coordinate
#     cv2.rectangle(img, (x, y), ((x+w), (y+w)), (0, 255, 0), 2)

# cv2.imshow('Face Detector', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

webcam = cv2.VideoCapture(-1)

while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for coordinate in face_coordinates:
        (x, y, w, h) = coordinate
        cv2.rectangle(frame, (x, y), ((x+w), (y+w)), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)
    
    if key==81 or key==113:
        break

webcam.release()