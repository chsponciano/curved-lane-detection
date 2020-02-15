import numpy as np
import cv2

def initialize():
    set_values([42,63,14,87])

def nothing(e):
    print(f'Trackbar change: {e}')

def set_values(values):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", values[0],50, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", values[1], 100, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", values[2], 50, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", values[3], 100, nothing)

def get_values():
    _wt = cv2.getTrackbarPos("Width Top", "Trackbars")
    _wb = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    _ht = cv2.getTrackbarPos("Height Top", "Trackbars")
    _hb = cv2.getTrackbarPos("Height Bottom", "Trackbars")

    return np.float32([(_wt / 100, _ht / 100), (1-(_wt / 100), _ht / 100),
                      (_wb / 100, _hb / 100), (1-(_wb / 100), _hb / 100)])
