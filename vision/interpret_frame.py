import cv2

# Get the center region of the frame
def get_center_region(frame, size=(100, 100)):
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    x1 = center_x - size[0] // 2
    y1 = center_y - size[1] // 2
    x2 = x1 + size[0]
    y2 = y1 + size[1]
    return (x1, y1, size[0], size[1])

# Detect keypoints in an image using ORB
def detect_keypoints(frame):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(frame, None)
    return kp, des

# Match keypoints between two frames
def match_keypoints(kp1, des1, kp2, des2):
    if des1 is None or des2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Track Mario's pose
def track_mario_pose(prev_frame, curr_frame):
    kp1, des1 = detect_keypoints(prev_frame)
    kp2, des2 = detect_keypoints(curr_frame)
    matches = match_keypoints(kp1, des1, kp2, des2)
    return kp1, matches

# Check if Mario's pose has changed
def has_pose_changed(prev_frame, curr_frame):
    if prev_frame is None:
        return False
    _, matches = track_mario_pose(prev_frame, curr_frame)
    return len(matches) < 5  # Fewer matches suggest significant pose change

# Detect moving objects
def detect_moving_objects(prev_frame, curr_frame):
    if prev_frame is None or curr_frame is None:
        return []  # Return empty list if there's no previous frame
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detect walkable area
def detect_walkable_area(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detect enemies based on pose change upon collision
def detect_enemies_on_pose_change(curr_frame, prev_frame, enemies):
    if has_pose_changed(prev_frame, curr_frame):
        return detect_moving_objects(prev_frame, curr_frame)
    return []