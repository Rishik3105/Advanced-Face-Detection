import cv2 as cv
import mediapipe as mp
import time

cap=cv.VideoCapture(0)
mpface=mp.solutions.face_detection
face=mpface.FaceDetection()
mpdraw=mp.solutions.drawing_utils
cTime=0
PTime=0
while True:
    success,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=face.process(imgRGB)
    print(results.detections)
    #print(f'success={success}')
    if results.detections:
        for id,detection in enumerate(results.detections):
            #qprint(id,detection)
            #print(detections.location_data.relative_bounding_box)
            #mpdraw.draw_detection(img,detection)  # uncomment this line if you want to see the landmarks on the face 
           bboxc=detection.location_data.relative_bounding_box
           ih,iw,ic=img.shape
           bbox=int(bboxc.xmin*iw),int(bboxc.ymin*ih),int(bboxc.width*iw),int(bboxc.height*ih)
           cv.rectangle(img,bbox,(255,0,255),2)
           cv.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
    cTime=time.time()
    fps=1/(cTime-PTime)
    Ptime=cTime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,255,0),3)
    cv.imshow('Image',img)
    if cv.waitKey(1) & 0XFF==ord('q'):  # selct q to exit
        break
