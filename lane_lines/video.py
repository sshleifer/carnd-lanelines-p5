import cv2
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
from lane_lines.line import Lane
from lane_lines.fit_poly import fit_poly, warp, Minv, draw_line
from lane_lines.sobel_utils import get_threshd_image


def is_valid_curve(last_l, last_r, new_l, new_r):
    chg = max(np.abs(last_l - new_l), np.abs(last_r-new_r))
    lr_gap = np.abs(new_l-new_r)
    if (chg > 10000) or (lr_gap > 10000) or (new_l > 1e5) or (new_r > 1e5):
        return False
    else:
        return True

def _put_text(output, txt, height):
    cv2.putText(output, txt, (100, height),
                cv2.FONT_HERSHEY_SIMPLEX,
                2, (0,0,0), 10)


l_curve = 3000
r_curve = 3000

def video_pipeline(clip):


    left_lane = Lane(curve=3000)
    right_lane = Lane(curve=3000)
    l_curve = 3000
    r_curve = 3000
    left_lane = {}
    logs = []
    usedl = []
    usedr = []
    imgs = []

    for k in clip.iter_frames():
        binary_warped = warp(get_threshd_image(k))
        big_tuple = fit_poly(binary_warped)
        left_fit, right_fit, left_curve, right_curve, offset = big_tuple[:5]
        logs.append([
            left_fit, right_fit, left_curve, right_curve, offset])

        overwrite = is_valid_curve(l_curve,
                                   r_curve,
                                   left_curve,
                                   right_curve)

        if overwrite:
            l_curve = left_curve
            r_curve = right_curve
            left_fit = left_fit
            right_fit = right_fit
            # left_lane = Lane(left_fit, left_curve)
            # right_lane = Lane(right_fit, right_curve)
            # usedl.append(left_lane)
            # usedr.append(right_lane)

        output = draw_line(k, left_fit, right_fit, Minv)
        _put_text(output, 'Left curvature: {:.0f} m'.format(l_curve),
                  50)
        _put_text(output, 'Right curvature: {:.0f} m'.format(r_curve),
                  100)
        _put_text(output, 'Vehicle is {:.2f}M right of center'.format(offset),
                  150)
        imgs.append(output)

    return imgs, logs