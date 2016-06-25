import cv2
import numpy as np

# Postavljanje putanje do fajla u kome se vrsi detektovanje
video_path = "test_video/TownCentreXVID.avi"

# Odredjivanje dimenzije frejma
frame_width = 450
frame_height = 250

def mark_people(gray, original):
    count_people = 0;
    ret, threshold = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', threshold)

    # threshold = cv2.dilate(threshold, np.ones((5, 5)))
    # threshold = cv2.erode(threshold, np.ones((5, 5)))
    threshold = cv2.erode(threshold, np.ones((3, 3)))
    # threshold = cv2.dilate(threshold, np.ones((3, 3)))

    contours_img, contours, hierarchy = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('contours', contours_img)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w>20 and h>20):
            cv2.rectangle(original, (x,y), (x+w, y+h), (0,0,255),2)
            count_people = count_people+1

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Count of people: '+str(count_people)
    cv2.putText(original, text, (10,10), font, 0.5, (255,0,0), 2,cv2.LINE_AA)
    cv2.imshow('Obelezeni ljudi', original)



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

while (cap.isOpened()):
    # Citanje drugog tj. treceg frejma i promena njegove dimenzije
    ret, frame2 = cap.read()
    ret, frame2 = cap.read()
    frame2 = cv2.resize(frame2, (frame_width, frame_height))
    frame2_org = np.copy(frame2)                                        # Originalni sledeci frejm (zbog prikaza)

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
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    cv2.imshow('Original', frame2_org)
    cv2.imshow('Detektovano rgb', rgb)
    cv2.imshow('Detektovano sivo', gray)
    mark_people(gray.copy(), frame2_org.copy())

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('opitcalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)

    prvs = next

cap.release()
cv2.destroyAllWindows()
