import cv2
import numpy as np
from PIL import Image 
cap=cv2.VideoCapture("crop-1.mp4")
from pytesseract import image_to_string
#img = cv2.imread("00039car.jpg")
plate_cascade = cv2.CascadeClassifier('./indian_license_plate.xml')

	# detects numberplates and returns the coordinates and dimensions of detected license plate's contours.


while True:
    ret, frame =cap.read()
    img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("gray", img)
    plate_rect = plate_cascade.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 7)
    #print(plate_rect)
    if(plate_rect is ()):
        continue
    for (x,y,w,h) in plate_rect:
        a,b = (int(0.0025*img.shape[0]), int(0.025*img.shape[1])) #parameter tuning
    plate_img = img[y:y+h-a, x:x+w+b]
    cv2.rectangle(img, (x,y), (x+w, y+h), (51,51,255), 3)
    img1= cv2.adaptiveThreshold(plate_img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
    img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
#img = image_smoothening(img)
    img1 = cv2.GaussianBlur(img1,(3,3),0)
#cv2.imshow('image',img)
    #cv2.imshow('processed',img1)
    print(image_to_string(img1, lang='eng'))
    cv2.imshow("plate_img", plate_img)
    cv2.waitKey(0)
    
    k=cv2.waitKey(20)&0xff
    if k==27:
        break
cv2.destroyAllWindows()
