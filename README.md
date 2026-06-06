# RL Car Project

This repository contains a simple reinforcement learning (RL) environment and training script using the Stable Baselines3 library. The goal is to train an agent to navigate around a custom track using Proximal Policy Optimization (PPO).

## Features

- Customizable track creation with a figure-eight shape.
- Option to run training with or without GUI for visualization.
- Uses Stable Baselines3 for PPO implementation.

## Requirements

To run this project, you need the following Python packages:

- `gymnasium`
- `stable-baselines3`
- `pygame`

You can install these dependencies using pip:

```sh
pip install gymnasium stable-baselines3 pygame
```

Make sure to have a compatible version of Python installed (preferably 3.7 or higher).

## Usage

### Training the Agent

To train the agent, run the `train.py` script with or without GUI visualization.

- **With GUI**:
  ```sh
  python train.py --gui
  ```

- **Without GUI**:
  ```sh
  python train.py
  ```

During training, you will see the car's progress in the window if running with GUI. The training process will continue until it reaches the specified number of timesteps (80,000 by default).

### Testing the Trained Agent

After training, you can test the agent using the `test.py` script.

```sh
python test.py
```

This script will load the trained model (`car_ppo`) and run the car in the environment to demonstrate its learned behavior.

## Project Structure

- **`car_env.py`**: Defines the custom RL environment for the car.
- **`train.py`**: Trains the PPO model on the `CarEnv`.
- **`test.py`**: Tests the trained agent using the `CarEnv`.
- **`README.md`**: This file.

## Notes

- Ensure that you have the `car.png` image in the project directory for better visualization. If not, a red rectangle will be used as a fallback.

Feel free to explore and modify the code to suit your needs!