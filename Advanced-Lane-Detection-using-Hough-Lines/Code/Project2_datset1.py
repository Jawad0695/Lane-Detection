# Importing all the needed libraries
import cv2
import glob
import numpy as np

# Finding Homography for the four corners in the ROI of the frame
pts_src = np.array([[565, 36], [724, 36], [994, 269], [178, 270]])
pts_dst = np.array([[50, 0], [250, 0], [250, 500], [0, 500]])
H_Matrix, status = cv2.findHomography(pts_src, pts_dst)


# Undistorting the image based on the camera parameters
def undistort_image(img):
    k = [[9.037596e+02, 0.000000e+00, 6.957519e+02], [0.000000e+00, 9.019653e+02, 2.242509e+02],
         [0.000000e+00, 0.000000e+00, 1.000000e+00]]
    k = np.array(k)
    d = [[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]]
    d = np.array(d)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(k, d, (w, h), 1, (w, h))
    dst = cv2.undistort(img, k, d, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


# Function to calculate mean of a list
def mean(list):
    return float(sum(list)) / max(len(list), 1)


# Function to detect the lanes from the given image
def detect_lanes(img):
    lower_mask_white = np.array([0, 220, 0], dtype='uint8')
    upper_mask_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(img, lower_mask_white, upper_mask_white)
    white_detect = cv2.bitwise_and(img, img, mask=mask_white).astype(np.uint8)
    return white_detect


# Function to extrapolate the lines and draw the bounding region between the lanes
def extrapolate(img, lines, color=[0, 255, 0], thickness=12):
    right = "right"
    left = "left"
    straight = "straight"

    if lines is None:
        return

    ymin_global = img.shape[0]
    ymax_global = img.shape[0]

    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []

    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)

            if gradient > 0:
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]

    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)

    if (len(all_left_grad) > 0) and (len(all_right_grad) > 0):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(img, (upper_left_x, ymin_global),
                 (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global),
                 (lower_right_x, ymax_global), color, thickness)

        pts = np.array([[upper_left_x, ymin_global], [upper_right_x, ymin_global], [lower_right_x, ymax_global],
                        [lower_left_x, ymax_global]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillConvexPoly(img, pts, (100, 100, 100), lineType=8, shift=0)
        if upper_right_x > 100:
            return right
        elif upper_right_x < 30:
            return left
        else:
            return straight


# Function to detect the Hough lines
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    direction = extrapolate(line_img, lines)
    return line_img, direction


# The loop to iterate between all the images in the given dataset
for img in glob.glob((r"C:\Users\prade\OneDrive\Documents\UMD_Robotics\Courses\ENPM 673 - Perception\Project2\data_1-20200307T183325Z-001\data_1\data\*.png")):
    # Reading the images
    cv_img = cv2.imread(img)

    # Creating an ROI rejecting the sky region
    crop_img = cv_img[235:512, 0:1392]

    # Undistorting the image
    nice_img = undistort_image(crop_img)

    # Detecting white lines by creating a mask
    white_img = detect_lanes(nice_img)

    # Smoothing the image by using a Gaussian Filter
    white_img = cv2.GaussianBlur(white_img, (5, 5), 0)

    # Warping the image to get the bird's view of the road
    im_out = cv2.warpPerspective(white_img, H_Matrix, (300, 600))

    # Detecting the edges in the image
    edges = cv2.Canny(im_out, 100, 200)
    edges[0:500, 220:300] = 0

    # Detecting the lines in the image by using Hough Transform
    rho = 2
    theta = np.pi / 180
    threshold = 100
    min_line_len = 150
    max_line_gap = 800
    α = 0.8
    β = 1
    λ = 0
    line_image, direction = hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)

    # Adding the lines to the warped image
    result = cv2.addWeighted(im_out, α, line_image, β, λ)

    # Rewarping the image after the image processing is done
    out = cv2.warpPerspective(result, np.linalg.inv(H_Matrix), (1328, 205))

    # Image Addition
    rows, cols, channels = out.shape
    roi = nice_img[0:rows, 0:cols]

    # Now create a mask of the lines and create its inverse mask
    img2gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # add a threshold
    ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(out, out, mask=mask_inv)

    # Adding the image containing the lines and the actual frame
    Final = cv2.add(img1_bg, img2_fg)
    cv2.putText(Final, direction, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Final_output", Final)
    cv2.waitKey(1)

cv2.destroyAllWindows()