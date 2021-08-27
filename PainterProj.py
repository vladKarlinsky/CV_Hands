import cv2
import numpy as np
import os
import HandTrackingModule as htm


# Helper functions

def isSelectionMode(lmList):
    index = False
    middle = False

    if lmList[8][2] < lmList[7][2]:
        index = True
    if lmList[12][2] < lmList[11][2]:
        middle = True

    return index and middle


def isPaintingMode(lmList):
    index = False
    middle = False

    if lmList[8][2] < lmList[7][2]:
        index = True
    if lmList[12][2] >= lmList[11][2]:
        middle = True

    return index and middle


folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

"""

overlay[index] --> header
0 --> blue selected
1 --> eraser selected
2 --> green selected
3 --> red selected
4 --> start screen

"""
header = overlayList[4]
cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 640)
drawColor = (0, 0, 0)
brushThickness = 50
eraserThickness = 50
prev_x, prev_y = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)
dec = htm.handDetector(detectionCon=0.85)
stopped_paint = True

while True:

    success, img = cap.read()

    # Flip image for convenience when painting
    img = cv2.flip(img, 1)

    # Find hand landmarks
    img = dec.findHands(img, draw=False)
    lmList = dec.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        print(x1)
        if isPaintingMode(lmList):
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)  # Symbol of mode - circle

            if (prev_x == 0 and prev_y == 0) or stopped_paint:  # Config of first line drawn
                prev_x, prev_y = x1, y1
                stopped_paint = False

            if drawColor == (0, 0, 0):  # Eraser selected
                cv2.line(img, (prev_x, prev_y), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (prev_x, prev_y), (x1, y1), drawColor, eraserThickness)
            else:  # Brush selected
                cv2.line(img, (prev_x, prev_y), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (prev_x, prev_y), (x1, y1), drawColor, brushThickness)

            prev_x, prev_y = x1, y1

        if isSelectionMode(lmList):
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)  # Symbol of mode - circle
            if y1 < 66:  # Checking if we are in header
                if 0 < x1 < 160:  # Eraser is selected
                    header = overlayList[1]
                    drawColor = (0, 0, 0)
                elif 160 < x1 < 290:  # Red is selected
                    header = overlayList[3]
                    drawColor = (0, 0, 255)
                elif 290 < x1 < 420:  # Blue is selected
                    header = overlayList[0]
                    drawColor = (255, 100, 0)
                elif 420 < x1 < 540:  # Green is selected
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
            stopped_paint = True

    #  Handling the painting on main painter canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)

    # Header setup
    img[0:66, 0:640] = header

    cv2.imshow("Painter", img)
    cv2.waitKey(1)
