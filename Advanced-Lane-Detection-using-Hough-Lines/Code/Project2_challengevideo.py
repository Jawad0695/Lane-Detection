import cv2
import numpy as np
import os
from collections import deque

# === Parameters ===
HSL_YELLOW_LOWER = np.array([20, 120, 70], dtype=np.uint8)
HSL_YELLOW_UPPER = np.array([45, 200, 255], dtype=np.uint8)
HSL_WHITE_LOWER = np.array([0, 200, 0], dtype=np.uint8)
HSL_WHITE_UPPER = np.array([255, 255, 255], dtype=np.uint8)

CANNY_LOW = 100
CANNY_HIGH = 200
GAUSS_KERNEL = (5, 5)

HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LEN = 100
HOUGH_MAX_LINE_GAP = 200

# === Homography ===
pts_src = np.array([[574, 11], [762, 9], [1100, 246], [162, 254]])
pts_dst = np.array([[50, 0], [250, 0], [250, 500], [0, 500]])
H_Matrix, _ = cv2.findHomography(pts_src, pts_dst)

# === Camera Calibration ===
K = np.array([
    [1.15422732e+03, 0.000000e+00, 6.71627794e+02],
    [0.000000e+00, 1.14818221e+03, 3.86046312e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
])
D = np.array([[-0.242565104, -0.0477893070, -0.00131388084, -0.0000879107779, 0.0220573263]])

# === History buffers ===
lane_history = deque(maxlen=10)
direction_history = deque(maxlen=5)
last_offset = 0  # Keep last known offset

def undistort_image(img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    dst = cv2.undistort(img, K, D, None, newcameramtx)
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]

def detect_lanes_mask(img):
    hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask_yellow = cv2.inRange(hsl_img, HSL_YELLOW_LOWER, HSL_YELLOW_UPPER)
    mask_white = cv2.inRange(hsl_img, HSL_WHITE_LOWER, HSL_WHITE_UPPER)
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
    masked_img = cv2.bitwise_and(hsl_img, hsl_img, mask=combined_mask)
    return cv2.cvtColor(masked_img, cv2.COLOR_HLS2BGR)

def extrapolate_and_draw(img, lines, color=(0, 255, 0), thickness=12):
    global last_offset

    if lines is None or len(lines) == 0:
        return direction_history[-1] if direction_history else "no_lines"

    left_lines, right_lines = [], []
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]

    for x1, y1, x2, y2 in lines[:, 0]:
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if abs(slope) < 0.5:  # Ignore almost horizontal lines
            continue
        ymin_global = min(ymin_global, y1, y2)
        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))

    if not left_lines or not right_lines:
        # If one lane missing, keep last known offset
        offset = last_offset
    else:
        def average_line_params(lines):
            xs = [p for l in lines for p in (l[0], l[2])]
            ys = [p for l in lines for p in (l[1], l[3])]
            return np.polyfit(xs, ys, 1)

        left_grad, left_int = average_line_params(left_lines)
        right_grad, right_int = average_line_params(right_lines)

        lane_history.append((left_grad, left_int, right_grad, right_int))
        avg_left_grad, avg_left_int, avg_right_grad, avg_right_int = np.mean(lane_history, axis=0)

        # Compute intersection with bottom of image
        bottom_y = ymax_global
        bottom_left_x = int((bottom_y - avg_left_int) / avg_left_grad)
        bottom_right_x = int((bottom_y - avg_right_int) / avg_right_grad)

        lane_width = bottom_right_x - bottom_left_x
        lane_mid_x = (bottom_left_x + bottom_right_x) / 2
        img_mid_x = img.shape[1] / 2

        offset = (lane_mid_x - img_mid_x) / lane_width  # normalized offset
        last_offset = offset

        # Draw lanes
        cv2.line(img, (bottom_left_x, bottom_y), (int((ymin_global - avg_left_int) / avg_left_grad), ymin_global), color, thickness)
        cv2.line(img, (bottom_right_x, bottom_y), (int((ymin_global - avg_right_int) / avg_right_grad), ymin_global), color, thickness)

    # Decision
    if abs(offset) < 0.05:  # within 5% of lane width
        direction = "straight"
    elif offset > 0.05:
        direction = "right"
    else:
        direction = "left"

    direction_history.append(direction)
    return max(set(direction_history), key=direction_history.count)

def process_frame(frame):
    roi_crop = frame[480:720, 0:1280]
    undistorted = undistort_image(roi_crop)
    lane_masked = detect_lanes_mask(undistorted)
    blurred = cv2.GaussianBlur(lane_masked, GAUSS_KERNEL, 0)
    bird_view = cv2.warpPerspective(blurred, H_Matrix, (300, 600))
    edges = cv2.Canny(bird_view, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LINE_LEN, maxLineGap=HOUGH_MAX_LINE_GAP)
    lane_overlay = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    direction = extrapolate_and_draw(lane_overlay, lines)
    lane_unwarped = cv2.warpPerspective(lane_overlay, np.linalg.inv(H_Matrix),
                                        (undistorted.shape[1], undistorted.shape[0]))
    final = cv2.addWeighted(undistorted, 1.0, lane_unwarped, 0.7, 0)
    cv2.putText(final, direction, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Final Lane Detection", final)

# === Run on Video ===
video_path = "/home/jawad/robotic_arm/src/Advanced-Lane-Detection-using-Hough-Lines/Dataset/data_2/challenge_video.mp4"

if not os.path.exists(video_path):
    print(f"Error: Video not found at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break
    process_frame(frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
