import cv2
import numpy as np

video_path = "test_video/TownCentreXVID.avi"

cap = cv2.VideoCapture(video_path)

ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (500, 300))

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while (True):
    ret, frame2 = cap.read()
    frame2 = cv2.resize(frame2, (500, 300))
    fr = np.copy(frame2)

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Detektovano', rgb)
    cv2.imshow('Original', fr)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('opitcalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)

    prvs = next

cap.release()
cv2.destroyAllWindows()
