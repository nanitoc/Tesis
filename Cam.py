import cv2 as cv


cap = cv.VideoCapture(3)

while(True):
    ret, frame = cap.read()
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

    cv.imshow('frame', rgb)
    if cv.waitKey(1) & 0xFF == ord('q'):
        out = cv.imwrite('Root\Prueba\Trampa.png', frame)
        break

cap.release()
cv.destroyAllWindows()