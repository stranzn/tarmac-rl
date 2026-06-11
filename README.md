# tarmac-rl

A deep reinforcement learning project where a car learns to drive around a recreation of the Monza circuit entirely through trial and error. A PPO agent controls steering and throttle, receiving rewards for making forward progress and penalties for going off-road.

## Demo

Train the agent, then watch it drive using the built-in menu:

```sh
python main.py
```

## Features

- Hand-modelled Monza circuit built with a custom `TrackBuilder` class
- PPO agent via Stable Baselines3 with a tuned reward function
- 12-dimensional observation space including lookahead signals so the car can anticipate corners
- Independent steering and throttle via `MultiDiscrete` action space
- Ghost replay system — records a snapshot every 5k timesteps and plays all attempts simultaneously so you can watch the agent improve over time
- Lap timer with best lap tracking
- Simple pygame menu as a unified entry point

## Requirements

```sh
pip install gymnasium stable-baselines3 pygame-ce numpy
```

Python 3.10 or higher recommended.

## Project Structure

| File | Purpose |
|---|---|
| `menu.py` | Main entry point — launch training, testing, or replay from here |
| `car_env.py` | Gymnasium environment — physics, observations, reward function |
| `track_builder.py` | Constructs the Monza circuit from straight and arc primitives |
| `train.py` | Trains the PPO model and records ghost replay snapshots |
| `test.py` | Loads a trained model and watches it drive |
| `replay.py` | Plays all ghost snapshots simultaneously, colour coded by training stage |
| `replay_callback.py` | SB3 callback that records episode snapshots during training |
| `stopwatch.py` | Lap timing logic |

## Usage

### Menu

The easiest way to run everything:

```sh
python menu.py
```

Options:
- **Train Model (GUI)** — trains with a live render window
- **Train Model (Headless)** — faster, no window
- **Test Trained Model** — watch the trained agent drive
- **Watch Ghost Replays** — all training snapshots playing simultaneously
- **Quit**

### Training

```sh
python train.py           # headless, faster
python train.py --gui     # with render window
```

Trains for 120,000 timesteps by default. Saves the model as `car_ppo.zip` and replay snapshots to `replays.npy`. A checkpoint is saved every 5k timesteps into `./checkpoints/`.

### Testing

```sh
python test.py
```

Loads `car_ppo.zip` and runs the agent continuously. Controls: `SPACE` to pause, `ESC` to quit.

### Ghost Replay

```sh
python replay.py
```

Plays all recorded snapshots at once — one ghost car per 5k timestep interval, colour coded from red (early training) to green (late training). Controls: `SPACE` to pause, `R` to restart, `ESC` to quit.

Requires `replays.npy` to exist — run training first.

## How It Works

### Observation Space (12 values)

| Index | Value |
|---|---|
| 0 | Speed / MAX_SPEED |
| 1-2 | sin/cos of heading |
| 3 | Signed lateral distance from track centre |
| 4-5 | sin/cos of current heading error |
| 6-7 | sin/cos of heading error to waypoint 5 steps ahead |
| 8-9 | sin/cos of heading error to waypoint 15 steps ahead |
| 10-11 | sin/cos of heading error to waypoint 30 steps ahead |

### Action Space

`MultiDiscrete([3, 3])` — steering and throttle are independent axes:
- Axis 0: steer left / straight / steer right
- Axis 1: coast / accelerate / brake

### Reward Function

- **+1.5 per waypoint** advanced (gated on alignment and forward direction)
- **+0.10 × speed × alignment** bonus for moving fast in the right direction
- **+0.5** for steering the correct way into an upcoming corner
- **+0.2** for braking before a sharp corner
- **−0.15** per step when nearly stationary
- **−quadratic edge penalty** scaling up hard near the track boundary
- **−(5.0 + speed × 1.5)** on going off-track — faster crashes penalised more

## Notes

- Ensure `car.png` in the project directory for a proper car sprite. A red rectangle is used as fallback.
- The observation space and action space have changed significantly from earlier versions — old saved models are not compatible.