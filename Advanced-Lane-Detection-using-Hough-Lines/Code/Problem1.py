import numpy as np
import cv2

# Load video
video = cv2.VideoCapture('/home/jawad/robotic_arm/src/Advanced-Lane-Detection-using-Hough-Lines/Dataset/Night Drive - 2689.mp4')
if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

while True:
    ret, frame = video.read()
    if not ret:
        # Restart video
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Resize
    image = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # Night enhancement
    image = adjust_gamma(image, gamma=1.6)

    # CLAHE on V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    hsv_clahe = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

  


    # Color masking with looser thresholds
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    lower_yellow = np.array([15, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    color_mask = cv2.bitwise_or(mask_white, mask_yellow)

    # Debug: see masked colors
    cv2.imshow("Color Mask", color_mask)

    # Edge detection
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_and(edges, color_mask)

    cv2.imshow("Edges", edges)

    # ROI
    masked_edges = region_of_interest(edges)
    cv2.imshow("Masked Edges", masked_edges)

    # Hough
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 25, minLineLength=40, maxLineGap=150)
    output = image.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if 20 < abs(angle) < 160:
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Lane Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
