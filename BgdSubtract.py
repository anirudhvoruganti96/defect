import numpy as np
import cv2
from PIL import Image

cap = cv2.VideoCapture('test.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*"DIB ")
video = cv2.VideoWriter('output.mp4', fourcc, 30,size)

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask=Image.fromarray(fgmask)
    fgmask=fgmask.rotate(180)
    fgmask=np.array(fgmask)

    #finding contours
    #ret,thresh = cv2.threshold(fgmask,127,255,0)
    image,contour,hie = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for c in contour:
        fgmask=cv2.drawContours(fgmask, [c], -1, (0,255,0), 3)
    video.write(fgmask)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
video.release()
cv2.destroyAllWindows()
