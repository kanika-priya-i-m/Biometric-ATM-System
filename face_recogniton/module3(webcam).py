#..........................Automatic Webcam/rspb to capture image...........................#
import os
import time
import cv2

cap = cv2.VideoCapture(0)
i=0
while(True):
    ret, frame = cap.read()
    cv2.imshow("imshow",frame)
    i+=1
    time.sleep(4)
    cv2.imwrite('Test/a.jpg', frame)
    f=open('readdata.txt','w')
    f.write('read')
    f.close()
    face = "capture image"
    print("camera is start:",face )
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




