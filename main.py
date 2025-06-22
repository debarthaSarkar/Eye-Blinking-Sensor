#45:28
#EAR-Eye Aspect Ratio
import cv2 #for processing video and image
import cvzone #smilified common opencv operation 
from cvzone.FaceMeshModule import FaceMeshDetector #face landmark detector using mediapipe
from cvzone.PlotModule import LivePlot
#initialize the web cam
video=cv2.VideoCapture(0)

detector=FaceMeshDetector(maxFaces=1)

#graph setting 940 * 560, range[20,60] invert to show blinks up
plotY = LivePlot(940,560,[20,60],invert=True)


idList=[22,23,24,26,110,157,158,159,160,161,130,243]
#to store the history of EAR
ratioList=[] #Stores short history of ear to smooth noise
blinkCount=0 #Total number of blinks detected
counter=0 #Prevents multiple counts for one blink
blinkThreshold=35 #if the eye ratio below this a blink is detected
color=(0,0,255) #text color
plotColor=(0,0,255) #line color
holdFrames=8 #frames to wait after detecting a blink 
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
        leftLeft=face[130]
        leftRight=face[243]

        cv2.line(img,leftUp,leftDown,(0,255,0),2)
        cv2.line(img,leftLeft,leftRight,(0,255,0),1)
    #1000 is width 800 is height
    img = cv2.resize(img,(940,560))
    

    #calculating Eye Aspect Ration EAR for blinking logic
    #lower ration means eye is closed or blinking

    #calculate vertical distance for Eye closing
    lengthvertical, _ =detector.findDistance(leftUp,leftDown)
    #calculate horizontal distance for Eye closing
    lengthhorizontal, _ =detector.findDistance(leftLeft,leftRight)
    ratio=int((lengthvertical/lengthhorizontal)*100)

    ratioList.append(ratio)

    if len(ratioList)>3:
        ratioList.pop(0)
    ratioAvg= sum(ratioList)/len(ratioList)

    if ratioAvg<blinkThreshold and counter==0:
        blinkCount+=1
        counter=1
        color=(0,255,0)
        plotColor=(0,255,0)
    elif ratioAvg>=blinkThreshold:
        plotColor=(0,255,0)

    #delay
    if counter!=0:
        counter+=1
        if counter>holdFrames:
            counter=0
            color=(255,0,255)

    cvzone.putTextRect(img, f'Blink Count: {blinkCount}', (50, 100), colorR=color)


    imgPlot=plotY.update(ratioAvg)
    imgStack = cvzone.stackImages([img,imgPlot],2,1)

    cv2.imshow("video",imgStack)
    if cv2.waitKey(1) == ord('q'):
        break
#when the loop ends release the webcam and destroy all windows
video.release()
cv2.destroyAllWindows()
