import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

# Inizialize the Video
cap = cv2.VideoCapture('Videos/vid (2).mp4')

# Create the color Finder object
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

# Variables
posList = []

while True:
    success, img = cap.read()
    # img = cv2.imread("Ball.png")
    img = img[0:900, :]

    # Find the Color Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # Find location of the Ball
    imgCountours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        posList.append(contours[0]['center'])

    for i, pos in enumerate(posList):
        cv2.circle(imgCountours, pos, 5, (0, 255, 0), cv2.FILLED)
        if i == 0:
            cv2.line(imgCountours, pos, pos, (0, 255, 0), 2)
        else:
            cv2.line(imgCountours, pos, posList[i-1], (0, 255, 0), 2)

    # Display
    imgCountours = cv2.resize(imgCountours, (0, 0), None, 0.7, 0.7)
    # cv2.imshow('Image', img)
    cv2.imshow('ImageColor', imgCountours)
    cv2.waitKey(100)
