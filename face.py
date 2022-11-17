import cv2 as cv
import numpy as np

face_detector1 = cv.CascadeClassifier('C:\\Users\wwwvs\\AppData\\Local\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\data\\haarcascade_forntalface_default.xml')
eye_detector1 = cv.CascadeClassifier('C:\\Users\\wwwvs\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\data\\haarcascade_eye.xml')

cap=cv.VideoCapture(0)

while(1):
    res,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face=face_detector1.detectMultiScale(gray)
    
    for (x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    cv.imshow("frame",frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
