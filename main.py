#45:28
import cv2 #for processing video and image
import cvzone #smilified common opencv operation 
from cvzone.FaceMeshModule import FaceMeshDetector #face landmark detector using mediapipe
#initialize the web cam
video=cv2.VideoCapture(0)

detector=FaceMeshDetector(maxFaces=1)

idList=[22,23,24,26,110,157,158,159,160,161]

while True:
    #read frame from video footage
    success,img=video.read()
    img=cv2.flip(img,1)
    #detect face landmark
    img,faces=detector.findFaceMesh(img,draw=False)
    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img,face[id],1,(0,0,255),cv2.FILLED)
        leftUp=face[159]
        leftDown=face[23]

        cv2.line(img,leftUp,leftDown,(0,255,0),2)
    #1000 is width 800 is height
    img = cv2.resize(img,(1000,800))
    cv2.imshow("video",img)
    if cv2.waitKey(1) == ord('q'):
        break
#when the loop ends release the webcam and destroy all windows
video.release()
cv2.destroyAllWindows()
