import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tensorflow
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('model/keras_model.h5','model/labels.txt')
offset = 20
img_size = 300
counter = 0 

folder = 'Data/C'
labels = ["A","B","C"]

while True:
    success,img = cap.read()
    imgOutput = img.copy()
    hands,img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        img_white = np.ones((img_size,img_size,3),np.uint8)
        img_crop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        h = img_crop.shape[0]
        w = img_crop.shape[1]

        aspectRatio = h/w

        if aspectRatio>1:
            k = img_size/h
            wCal = math.ceil(w*k)
            img_resize = cv2.resize(img_crop,(wCal,img_size))
            wGap = math.ceil((img_size-wCal)/2)
            img_white[:,wGap:wGap+wCal] = img_resize
            prediction,idx = classifier.getPrediction(img_white)
            #print(prediction,idx)

        else:
            k = img_size/w
            hCal = math.ceil(h*k)
            img_resize = cv2.resize(img_crop,(img_size,hCal))
            hGap = math.ceil((img_size-hCal)/2)
            img_white[hGap:hGap+hCal,:] = img_resize
            prediction,idx = classifier.getPrediction(img_white)
            #print(prediction,idx)

        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+100,y-offset-50+50),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput,labels[idx],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

        cv2.imshow("image cropped",img_crop)
        cv2.imshow("image white",img_white)

    cv2.imshow("image",imgOutput)
    cv2.waitKey(1)

