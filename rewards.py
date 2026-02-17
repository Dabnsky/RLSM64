import numpy as np
import cv2

prev_frame = None

def get_movement_reward(current_frame):
    global prev_frame
    if prev_frame is None:
        prev_frame = current_frame
        return 0  # No reward on the first frame

    # Compute pixel difference between frames
    diff = cv2.absdiff(prev_frame, current_frame)
    movement_score = np.sum(diff) / 255.0  # Normalize

    prev_frame = current_frame  # Update previous frame
    return movement_score * 0.1  # Small reward for movement


def detect_coins(frame):
    if len(frame.shape) == 2 or frame.shape[0] == 1:  # If grayscale (1, H, W)
        frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)  # Convert to 3-channel BGR

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    coin_count = np.sum(mask) / 255  # Count yellow pixels
    return coin_count * 0.5  # Reward for each coin detected


def check_death(frame):
    brightness = np.mean(frame)
    if brightness < 20:  # Dark screen → Possible death
        return -10  # Big penalty
    return 0


def get_reward(state):
    movement_reward = get_movement_reward(state)
    coin_reward = detect_coins(state)
    death_penalty = check_death(state)

    return movement_reward + coin_reward + death_penalty
