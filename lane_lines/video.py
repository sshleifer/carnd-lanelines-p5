import cv2
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import matplotlib.pyplot as plt

from lane_lines.line import Lane
from lane_lines.fit_poly import fit_poly, warp, Minv, draw_line
from lane_lines.sobel_utils import get_threshd_image


def plot_gray(img, cmap='gray'):
    fig, axes = plt.subplots(ncols=1, figsize=(10, 10))
    axes.imshow(img, cmap=cmap)
    axes.axis('off')
    return axes

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



def show_diagnostic(big_tuple, img):
    (left_fit, right_fit,
            left_curverad, right_curverad, offset,
            left_lane_inds, right_lane_inds,
            out_img,  nonzerox, nonzeroy) = big_tuple
    return out_img
    # ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # ax = plot_gray(out_img, cmap=None)
    # ax.plot(left_fitx, ploty, color='yellow')
    # ax.plot(right_fitx, ploty, color='yellow')
    # return ax

def video_pipeline(clip):


    left_lane = Lane(curve=3000)
    right_lane = Lane(curve=3000)
    l_curve = 3000
    r_curve = 3000
    logs = []
    usedl = []
    usedr = []
    imgs = []
    diag = []

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
            left_lane = Lane(left_fit)
            right_lane = Lane(right_fit, right_curve)
            # usedl.append(left_lane)
            # usedr.append(right_lane)
        diag.append(show_diagnostic(big_tuple, k))
        output = draw_line(k, left_lane.fit, right_lane.fit, Minv)
        _put_text(output, 'Left curvature: {:.0f} m'.format(l_curve),
                  50)
        _put_text(output, 'Right curvature: {:.0f} m'.format(r_curve),
                  100)
        _put_text(output, 'Vehicle is {:.2f}M right of center'.format(offset),
                  150)
        imgs.append(output)

    return imgs, diag