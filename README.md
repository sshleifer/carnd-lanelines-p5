## Finding Lane Lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_output1.jpg "Undistorted Road"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "p5.ipynb" under the first header and follows closely from the lecture. We assume object points are the corners of the chessboard in the real world (so 0,0,0, for example), and then, for each chessboard image, store the imagepoints, the result of `cv2.findChessboardCorners` We then call `calibrateCamera` to learn a mapping from object points to image points, and then `cv2.undistort` to use the mapping to remove the distortion from other images. The result is below:
![alt text][image1]
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
![alt text][image2]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

lane_lines.sobel_utils.get_threshd_image at line 51, contains the final filter I used, all other filters are in that module. The code borrows heavily from the lecture but deletes some lines by making numpy masks in one step with the `apply_threshold` helper.


The preprocesisng pipeline works as follows:
  - convert to LHS and extract S channel
  - use cv2.sobel to filter to pixels where the x or y derivate is between 10 and 160
    - (implemented but did not use magnitude and direction threshold)
![alt text][image3]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is run by the `warp` function but leverages `M` and `Minv` created in the notebook
The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points by looking at an image:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 455      | 200, 0        |
| 705, 455      | 1080, 0       |
| 1130, 720     | 1080, 720     |
| 190, 720      | 1080, 0       |

The lane lines do, in fact look reasonably straight after warping:
![alt text][image4]
#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
![alt text][image5]
First, we choose the starting columns for a window by looking at the columns of the binary image with the largest sums to the left and right of the middle of the image, and creating of 100 pixels around that peak column. (we assume car currently in lane :))

For each side,  starting from the first (lowest green) box,
we collect the (y,x) pairs that are nonzero and in the box. This is our line. We then start the next box at the average x point in the image.
We collect points and build 9 boxes up the image in this way. And then estimate the true position of the lane line by fitting a 2 degree polynomial to all collected points. The resulting polynomial is indicated by the yellow lines.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.


Curvature is estimated as in the lecture notes
```{py}
def get_curvature(a, b, y):
    numer = (1 + (2*a*y+b)**2)**1.5
    denom = np.abs(2*a)
    return numer / denom
```

We find the vehicle's position in the lane by assuming it is in the middle of the image, and then seeing where in the image the lane lines are. Implementation is in `lane_lines/fit_poly.py` `get_offset` function.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I copied code to draw lines on the image from the lecture.
![alt text][image6]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./my_project_video.mp4)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  

My implementation follows fairly directly from the lecture notes, with the only large discrepancies being only using the x and y gradient thresholds and coding the numpy masks more cleanly. The latter deviation caused my largest bug, where warping the binary image was causing my jupyter kernel to crash because the mask was `int64` instead of `uint8`.

#### Where will your pipeline likely fail?  

The pipeline's current state is that it does not fail catastrophically in the first video, but wobbles around second 43. Implementing outlier rejection and/or smoothing did not initially help, presumably, but inspecting the binary_warped image at this time step (and for most of the challenge video) revealed utter chaos. It is likely to fail when that car is in one of the middle lanes, or the beginning of a lane line is obfuscated (throwing off the second window start).

#### What could you do to make it more robust?
Next steps that might fix the wobble and improve performance on the challenge video include:
- restrict histogram peak search (do not look for windows far from center
- use gradient direction and magnitude filters (already implemented!)
