import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#  Config of screen & camera
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0  # Used for fps calculation

detector = htm.handDetector(detectionCon=0.8)

#  Config of volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        if length < 30:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # Hand range 30 - 300
        # Volume range (-65) - 0

        vol = np.interp(length, [30, 300], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

        # Volume bar range 150 - 400
        barHeight = np.interp(vol, [minVol, maxVol], [400, 150])
        cv2.rectangle(img, (50, int(barHeight)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, 'Volume Control', (50, 435), cv2.FONT_HERSHEY_DUPLEX, 1,
                (255, 0, 0), 2)

    # FPS on screen
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 1,
                (255, 0, 0), 2)

    cv2.imshow("Volume Hand Control", img)
    cv2.waitKey(1)
