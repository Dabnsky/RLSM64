from stable_baselines3 import PPO
from env.mario_env import MarioEnv

env = MarioEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

model.save("mario_rl_model")

model = PPO.load("mario_rl_model")

obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()  # Show the frames for debugging