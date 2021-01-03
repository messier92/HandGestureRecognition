# https://www.mygreatlearning.com/blog/real-time-face-detection/
# Very bad results for mask - something to do with the tensorflow model perhaps?
# https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
import numpy as np
import cv2
import sys
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture frames from a camera 
cap = cv2.VideoCapture(0) 

while(1):
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        a3 = np.array( [[[x,y],[x,y+h],[x+w,y+h],[x+w,y]]], dtype=np.int32 )
        cv2.fillPoly(img, a3, 0 )


    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
