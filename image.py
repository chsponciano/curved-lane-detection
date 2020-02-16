import numpy as np
import cv2
from setting import *


def treat_frame(frame):
    _gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _blur = cv2.GaussianBlur(_gray, (3, 3), 5)
    return cv2.Canny(_blur, 50, 100)

def _marked(bgr):
    return cv2.inRange(bgr, np.array(WHITE_COLOR[0]), np.array(WHITE_COLOR[1])), cv2.inRange(bgr, np.array(YELLOW_COLOR[0]), np.array(YELLOW_COLOR[1]))

def _filter_color(frame):
    _hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _marked_white, _marked_yellow = _marked(_hsv)
    return cv2.bitwise_or(_marked_white, _marked_yellow)

def thresholding(frame, _ksize=(5,5), _iterations=1):
    _treat_frame = treat_frame(frame)
    _dilate_frame = cv2.dilate(_treat_frame, _ksize, iterations=_iterations)
    _erode_frame = cv2.erode(_dilate_frame, _ksize, iterations=_iterations)
    _color_frame = _filter_color(frame)
    return cv2.bitwise_or(_color_frame, _erode_frame), _treat_frame, _color_frame

def perpective(img, dst_size=(1280, 720), src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]), dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def inv_perspective_warp(img,
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def draw_points(img,src):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    #src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    src = src * img_size
    for x in range( 0,4):
        cv2.circle(img,(int(src[x][0]),int(src[x][1])),15,(0,0,255),cv2.FILLED)
    return img

def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 1 / img.shape[0]  # meters per pixel in y dimension
    xm_per_pix = 0.1 / img.shape[0]  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters

    return (l_fit_x_int, r_fit_x_int, center)
    
def draw_lanes(img, left_fit, right_fit,frameWidth,frameHeight,src):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))
    inv_perspective = inv_perspective_warp(color_img,(frameWidth,frameHeight), dst=src)
    inv_perspective = cv2.addWeighted(img, 0.5, inv_perspective, 0.7, 0)
    return inv_perspective

def draw_line(img,lane_curve):
    myWidth = img.shape[1]
    myHeight = img.shape[0]
    # print(myWidth,myHeight)
    for x in range(-30, 30):
        w = myWidth // 20
        cv2.line(img, (w * x + int(lane_curve // 100), myHeight - 30),
                 (w * x + int(lane_curve // 100), myHeight), (0, 0, 255), 2)
    cv2.line(img, (int(lane_curve // 100) + myWidth // 2, myHeight - 30),
             (int(lane_curve // 100) + myWidth // 2, myHeight), (0, 255, 0), 3)
    cv2.line(img, (myWidth // 2, myHeight - 50), (myWidth // 2, myHeight), (0, 255, 255), 2)

    return img