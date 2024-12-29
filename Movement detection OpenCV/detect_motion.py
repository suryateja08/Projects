import cv2 as s
import numpy as np

cap=s.VideoCapture('vtest.avi')

ret,frame1=cap.read()
ret,frame2=cap.read()

while cap.isOpened():
    diff=s.absdiff(frame1,frame2)
    gray=s.cvtColor(diff, s.COLOR_BGR2GRAY)
    _,thresh=s.threshold(gray,20,255,s.THRESH_BINARY)
    #dilated=s.dilate(thresh,None,iterations=3)
    dilated=s.morphologyEx(thresh,s.MORPH_GRADIENT,kernel=np.ones((5,5),np.uint8))
    contours,_=s.findContours(dilated,s.RETR_TREE,s.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x,y,w,h)=s.boundingRect(contour)

        if s.contourArea(contour)<500:
            continue
        s.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        s.putText(frame1,"Status:{}".format('movement'),(10,20),s.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    s.imshow("feed",frame1)
    frame1=frame2
    ret,frame2=cap.read()

    if s.waitKey(40)==27:
        break
s.destroyAllWindows()
cap.release()
