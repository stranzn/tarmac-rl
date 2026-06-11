import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from car_env import CarEnv

REPLAY_FILE    = "replays.npy"
RECORD_EVERY   = 5_000   # timesteps between snapshots

class ReplayCallback(BaseCallback):
    """
    Every RECORD_EVERY timesteps, runs one full episode with the current
    policy and saves the (x, y, angle) trajectory to REPLAY_FILE.
    """
    def __init__(self):
        super().__init__()
        self._next_record_at = RECORD_EVERY
        self._replays        = []   # list of np.ndarray, each shape (T, 3)

        # load any existing replays so training can be resumed without losing history
        if os.path.exists(REPLAY_FILE):
            existing = np.load(REPLAY_FILE, allow_pickle=True).tolist()
            self._replays = existing
            print(f"[ReplayCallback] Loaded {len(existing)} existing replays from {REPLAY_FILE}")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_record_at:
            self._record_episode()
            self._next_record_at += RECORD_EVERY
        return True

    def _record_episode(self):
        # Run one episode with the current policy and store the trajectory
        env = CarEnv(render_mode=None)
        obs, _ = env.reset()
        frames  = []
        done    = False
        truncated = False

        while not done and not truncated:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            frames.append((env.x, env.y, env.angle))

        env.close()

        trajectory = np.array(frames, dtype=np.float32)  # shape (T, 3)
        self._replays.append(trajectory)

        # persist immediately so a crash mid-training doesn't lose data
        np.save(REPLAY_FILE, np.array(self._replays, dtype=object))
        print(f"[ReplayCallback] Saved replay #{len(self._replays)} "
              f"at {self.num_timesteps} timesteps ({len(frames)} frames)")