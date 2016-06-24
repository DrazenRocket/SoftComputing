import cv2
import numpy as np

# Postavljanje putanje do fajla u kome se vrsi detektovanje
video_path = "test_video/TownCentreXVID.avi"

# Odredjivanje dimenzije frejma
frame_width = 450
frame_height = 250

# Inicijalizacija rada sa videom (otvaranje videa)
cap = cv2.VideoCapture(video_path)

# Citanje prvog frejma i promena njegovih dimenzija
ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (frame_width, frame_height))

# Pretvaranje prvog frejma u sivi
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Formiranje hsv pomocu kog ce se prikazati rezultat
hsv = np.zeros_like(frame1)                 # np.zeros_like(frame1) vraca niz istih dimenzija kao frame1 ali sa nulama
hsv[..., 1] = 255                           # green

while (True):
    # Citanje drugog tj. treceg frejma i promena njegove dimenzije
    ret, frame2 = cap.read()
    ret, frame2 = cap.read()
    frame2 = cv2.resize(frame2, (frame_width, frame_height))
    fr = np.copy(frame2)                                        # Originalni sledeci frejm (zbog prikaza)

    # Pretvaranje drugog tj. treceg frejma u sivi
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Racunaje flow-a pomocu Farneback-ovog algoritma
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Odredjivanje magnitude i ugla vektora (datai su x i y)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Konverzija ugla i normalizacija magnitude vektora
    hsv[..., 0] = ang*180/np.pi/2                       # sa dva zbog opsega
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Konvertovanje hsv u boju
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Original', fr)
    cv2.imshow('Detektovano', rgb)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('opitcalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)

    prvs = next

cap.release()
cv2.destroyAllWindows()
