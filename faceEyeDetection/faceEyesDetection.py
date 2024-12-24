import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 1.3 is the scaling factor i.e how slow to resize the input img and rerun the algo to get accurate result and 5 is the Parameter specifying how many neighbors each candidate rectangle should have to retain it.This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.

    for (x,y,w,h) in faces :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        regionOfInterest = gray[y:y+h, x:x+w]
        regionOfInterestColor = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(regionOfInterest, 1.3,5)
        for (ex,ey,ew,eh) in eyes :
            cv2.rectangle(regionOfInterestColor,(ex,ey), (ex+ew,ey+eh), (0,255,255), 5 )

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()