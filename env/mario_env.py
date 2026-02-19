from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import gym
from gym import spaces
import mss
import cv2
import numpy as np
from pynput.keyboard import Controller
import time
import rewards
import pygetwindow as gw

# Define the region where the emulator is displayed (adjust as needed)
EMULATOR_REGION = {"top": 100, "left": 100, "width": 640, "height": 480}



# Define the action keys
ACTIONS = ["w", "a", "s", "d", ",", "l", "k"]  # Move Forward, Left, Back, Right, Jump
kb = Controller()


class MarioEnv(gym.Env):
    def __init__(self):
        self.game_window = None  # Store the game window
        self.sct = mss.mss()
        self.monitor = self.get_game_window()
        self.object_disappearance_count = {}  # Track how long objects have been missing
        super(MarioEnv, self).__init__()
        self.action_space = spaces.Discrete(7)  # 5 actions (up, left, right, down, jump)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, 84, 84), dtype=np.float32)
        self.visited_positions = set()  # Track explored positions


        # Previous frame storage for movement tracking
        self.prev_frame = None
        self.prev_pose = None

    def get_center_region(self, frame, size=(100, 100)):
        """Returns the bounding box (x, y, width, height) of the center region."""
        if frame is None or not hasattr(frame, 'shape'):
            raise ValueError("❌ Error: Invalid frame passed to get_center_region()")

        if len(frame.shape) == 2:  # Grayscale image
            h, w = frame.shape
        else:  # Color image
            h, w, _ = frame.shape

        center_x, center_y = w // 2, h // 2
        x1 = max(center_x - size[0] // 2, 0)
        y1 = max(center_y - size[1] // 2, 0)
        w, h = size

        return x1, y1, w, h  # ✅ Return coordinates, NOT an image


    def detect_keypoints(self, frame):
        """Detect keypoints in an image using ORB."""
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(frame, None)
        return kp, des
    
    # Match keypoints between two frames
    def match_keypoints(self, kp1, des1, kp2, des2):
        if des1 is None or des2 is None:
            return []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def capture_frame(self):
        """ Captures the game window. """
        frame = np.array(self.sct.grab(self.monitor))[:, :, :3]  # Drop alpha channel
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame = cv2.resize(frame, (84, 84))  # Resize for RL training

        return frame

    def has_pose_changed(self, prev_pose, curr_pose):
        """ Checks if Mario's pose has changed. """
        if prev_pose is None or curr_pose is None:
            return False
        return not np.array_equal(prev_pose, curr_pose)  # Compare feature vectors

    # Detect moving objects
    def detect_moving_objects(self, prev_frame, curr_frame):
        if prev_frame is None or curr_frame is None:
            return []  # Return empty list if there's no previous frame
        diff = cv2.absdiff(prev_frame, curr_frame)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_positions = [cv2.boundingRect(cnt) for cnt in contours]  # Track objects
        return object_positions


    def ensure_grayscale(self, frame):
        """Convert frame to grayscale if it's not already."""
        if len(frame.shape) == 3:  # If it has 3 channels (RGB/BGR)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame  # Already grayscale, return as is

    # Detect walkable area
    def detect_walkable_area(self, frame):
        gray = self.ensure_grayscale(frame)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def is_position_walkable(self, grid_x, grid_y, walkable_area):
        """Check if a given grid position is part of the walkable area."""
        if walkable_area is None:
            return False

        for contour in walkable_area:
            for point in contour:
                x, y = point[0]
                if abs(x // 10 - grid_x) < 1 and abs(y // 10 - grid_y) < 1:
                    return True
        return False

        # Detect enemies based on pose change upon collision
    def detect_enemies_on_pose_change(curr_frame, prev_frame, enemies):
        if self.has_pose_changed(prev_frame, curr_frame):
            return self.detect_moving_objects(prev_frame, curr_frame)
        return []

    def track_mario_pose(self, prev_frame, curr_frame):
        """Track Mario's pose using keypoints."""
        kp1, des1 = self.detect_keypoints(prev_frame)
        kp2, des2 = self.detect_keypoints(curr_frame)
        matches = self.match_keypoints(kp1, des1, kp2, des2)
        return kp1, matches

    def reset(self):
        """ Resets the environment. """
        self.prev_frame = None
        self.prev_pose = None
        frame = self.capture_frame()
        return np.expand_dims(frame, axis=0)  # Reshape to (1, 84, 84)


    def step(self, action):
        """ Steps through the environment based on the given action. """
        self.press_key(action)  # Send action to game

        # Capture the new frame
        curr_frame = self.capture_frame()
        print(f"✅ Step() - Frame shape: {curr_frame.shape}, Type: {type(curr_frame)}")
        
        if curr_frame is None:
            raise ValueError("❌ capture_frame() returned None!")
        print(f"✅ Captured frame shape: {curr_frame.shape}")  # Should be (height, width) or (height, width, 3)

        # Detect walkable area
        walkable_contours = self.detect_walkable_area(curr_frame)
        
        # Get Mario's position
        print(f"Captured frame shape: {curr_frame.shape if curr_frame is not None else 'None'}")
        mario_x, mario_y, mario_w, mario_h = self.get_center_region(curr_frame)

        # Check if Mario is inside any walkable contour
        inside_walkable = any(cv2.pointPolygonTest(cnt, (mario_x, mario_y), False) >= 0 for cnt in walkable_contours)

        # Extract Mario's pose (ORB keypoints at center)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(curr_frame, None)
        mario_pose = descriptors if descriptors is not None else np.array([])

    
        reward = self.calculate_reward(curr_frame)

        if inside_walkable:
            reward += 1  # Reward for moving into a walkable area
        else:
            reward -= 1  # Penalty for stepping outside walkable area

        # Update previous frame & pose
        self.prev_frame = curr_frame.copy()
        self.prev_pose = mario_pose

        # Environment termination condition (customize as needed)
        done = False

        # Reshape observation for Stable-Baselines3 (1, 84, 84)
        return np.expand_dims(curr_frame, axis=0), reward, done, {}


    def calculate_reward(self, curr_frame):
        """Compute rewards based on game state."""
        reward = 0  

        if self.prev_frame is not None:
            prev_objects = self.detect_moving_objects(self.prev_frame, curr_frame)
            curr_objects = self.detect_moving_objects(curr_frame, curr_frame)

            # Find objects that were in prev_objects but are missing from curr_objects
            disappeared_objects = [obj for obj in prev_objects if obj not in curr_objects]


            if disappeared_objects:
                reward += 10  # Reward when objects (e.g., coins or enemies) disappear

            # Get Mario's bounding box (center region)
            mario_x, mario_y, mario_w, mario_h = self.get_center_region(curr_frame)

            # Check if Mario's pose changed
            pose_changed = self.has_pose_changed(self.prev_frame, curr_frame)

            new_tracked_objects = []

            for obj in prev_objects:
                x, y, w, h = obj

                # Ignore objects inside Mario (particles clipping through him)
                if mario_x < x < mario_x + mario_w and mario_y < y < mario_y + mario_h:
                    continue  # Ignore this object

                # If object is not in the current frame, check if it was missing in previous frames too
                if obj not in curr_objects:
                    self.object_disappearance_count[obj] = self.object_disappearance_count.get(obj, 0) + 1
                    if self.object_disappearance_count[obj] > 2 and not pose_changed:
                        disappeared_objects += 1  # Only count objects that have been gone for 2+ frames
                else:
                    self.object_disappearance_count[obj] = 0  # Reset disappearance counter

                new_tracked_objects.append(obj)

            # Update the tracking memory
            self.tracked_objects = new_tracked_objects
            
        self.prev_frame = curr_frame  # Update previous frame


        # Detect walkable area
        walkable_area = self.detect_walkable_area(curr_frame)

        # Get Mario's current position
        mario_x, mario_y, _, _ = self.get_center_region(curr_frame)

        # Convert position to a grid cell (for tracking exploration)
        grid_x, grid_y = mario_x // 10, mario_y // 10  

        # Ensure Mario is in a walkable region before rewarding exploration
        if self.is_position_walkable(grid_x, grid_y, walkable_area):
            if (grid_x, grid_y) not in self.visited_positions:
                self.visited_positions.add((grid_x, grid_y))
                reward += 5  # Encourages exploration

        return reward

    def render(self, mode="human"):
        """ Optional rendering of the game frame (for debugging). """
        frame = self.capture_frame()
        cv2.imshow("MarioEnv", frame)
        cv2.waitKey(1)


    def close(self):
        """ Cleanup function. """
        cv2.destroyAllWindows()
 

    def press_key(self, action):
        """Presses the mapped key."""
        key = ACTIONS[action]
        kb.press(key)
        time.sleep(0.1)  # Adjust duration if needed
        kb.release(key)
    

    # Find the game window
    game_window = gw.getWindowsWithTitle("sm64coopdx v1.0.4")[0]  # Change to match exact title
    print(f"Game window found: {game_window.title}")


    def get_game_window(self):
        """Find and return the bounding box of the Super Mario 64 Co-Op Deluxe window."""
        windows = gw.getWindowsWithTitle("sm64coopdx v1.0.4")  # Adjust title if needed
        if not windows:
            raise Exception("❌ Game window not found! Make sure it's running.")

        win = windows[0]  # Get the first matching window
        bbox = {
            "top": win.top,
            "left": win.left,
            "width": win.width,
            "height": win.height
        }
        print(f"✅ Game window found at {bbox}")
        return bbox  # Return the bounding box