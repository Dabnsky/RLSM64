# RLSM64: Reinforcement Learning for Mario 64

## Project Overview
RLSM64 is a reinforcement learning project that uses computer vision to train an AI agent to play Super Mario 64. The agent interacts with the game through a custom OpenAI Gym environment, leveraging frame analysis and reward shaping to encourage exploration, object collection, and path following.

## Key Features
- **Computer Vision:** Uses OpenCV to process game frames, detect movement, coins, and Mario's pose.
- **Reinforcement Learning:** Employs Stable Baselines3 (PPO/DQN) for training the agent.
- **Custom Environment:** Modular `MarioEnv` class for interfacing with the game and managing state.
- **Reward System:** Rewards for movement, collecting coins, exploring new areas, and following paths; penalties for death or leaving walkable areas.
- **Exploration Tracking:** Tracks visited positions on a fine grid to encourage thorough exploration.
- **Path Following (Goal):** Ongoing improvements to help the agent recognize and follow level paths for better progress.

## Folder Structure
```
RLSM64/
├── env/         # Custom environment and wrappers
├── vision/      # Computer vision utilities
├── rewards/     # Reward calculation logic
├── scripts/     # Training and evaluation scripts
├── config/      # Config files (YAML/JSON)
├── tests/       # Unit and integration tests
```

## Goals
- Improve the agent's ability to make progress and follow intended level paths.
- Refine reward shaping to better distinguish between meaningful exploration and random movement.
- Modularize code for maintainability and extensibility.

## Getting Started
1. Install dependencies from `requirements.txt`.
2. Run training with `python scripts/train_rl_mario.py`.
3. Adjust environment, vision, and reward logic as needed for new levels or behaviors.

## Future Work
- Enhance path detection and following using higher resolution and finer grid tracking.
- Add configuration files for easy parameter tuning.
- Expand test coverage and add evaluation scripts.

---
For questions or contributions, open an issue or pull request.
