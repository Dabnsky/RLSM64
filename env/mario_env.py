from stable_baselines3 import DQN, PPO
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

ACTIONS = ["w", "a", "s", "d", ",", "l", "k"]  # Move Forward, Left, Back, Right, Jump
kb = Controller()

class MarioEnv(gym.Env):
    # ...existing code from LearnMario.py MarioEnv class...
    pass