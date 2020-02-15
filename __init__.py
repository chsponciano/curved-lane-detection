import numpy as np
import cv2
from setting import *
from controller import *
from image import *
import trackbar

def execute(capture, content_curves, index=0, _process=True):
    
    while _process:
        _, _frame = capture.read()
        try:
            _frame = cv2.resize(_frame, DIMENSION_FRAME, None)
        except:
            print('Simulation completed')

        _final_frame = _frame.copy()

        _undistort_frame = undistort(_frame, PATH_PICKLE)
        _thres_frame, _canny_frame, _color_frame = thresholding(_undistort_frame)    
        _src = trackbar.get_values()
        _warp_frame = perpective(_thres_frame, dst_size=DIMENSION_FRAME, src=_src)
        _warp_points_frame = draw_points(_frame.copy(), _src)
        
        _sliding_frame, _current_curves, _, _ = sliding_window(_warp_frame, draw_windows=True)

        try:
            _curverad = get_curve(_final_frame, _current_curves[0], _current_curves[1])
            _lane_curve = np.mean([_curverad[0], _curverad[1]])
            _final_frame = draw_lanes(_frame, _current_curves[0], _current_curves[1], DIMENSION_FRAME[0], DIMENSION_FRAME[1], src=_src)

            _curve = _lane_curve // 50

            if  int(np.sum(content_curves)) == 0:
                averageCurve = _curve
            else:
                averageCurve = np.sum(content_curves) // content_curves.shape[0]

            if abs(averageCurve -_curve) > 200: 
                content_curves[index] = averageCurve
            else:
                content_curves[index] = _curve

            index += 1

            if index >= POINT_LIMIT: 
                index = 0

            cv2.putText(_final_frame, str(int(averageCurve)), (DIMENSION_FRAME[0] // 2 - 70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2, cv2.LINE_AA)

        except Exception as ex:
            print(ex)
            _lane_curve = 0

        _final_frame= draw_line(_final_frame, _lane_curve)
        _thres_frame = cv2.cvtColor(_thres_frame,cv2.COLOR_GRAY2BGR)
        _blank_frame = np.zeros_like(_frame)

        _pipeline = stack_images(0.7, ( [_frame, _undistort_frame, _warp_points_frame],
                                        [_color_frame, _canny_frame, _thres_frame],
                                        [_warp_frame, _sliding_frame, _final_frame]))

        cv2.imshow("PipeLine Monitoring", _pipeline)
        cv2.imshow("Automation", _final_frame)

        if cv2.waitKey(1) == 27:
            _process = False
                

if __name__ == "__main__":
    _capture = cv2.VideoCapture(PATH_VIDEO)
    _curves = np.zeros([POINT_LIMIT])
    
    trackbar.initialize()
    
    execute(_capture, _curves)
    
    cv2.destroyAllWindows()
    _capture.release()
