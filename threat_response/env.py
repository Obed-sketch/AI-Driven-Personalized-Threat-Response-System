"""Step Two of the original script: the custom Gym environment.

Same semantics as the original SecurityEnv: stepping through the dataset one
row at a time; reward is +1 if that row's threat was resolved, -1 otherwise,
plus the row's normalized feedback. The only change is the gym>=0.26 API:
reset() returns (observation, info) and step() returns a 5-tuple.
"""
import gym
import numpy as np
from gym import spaces

from .preprocessing import FEATURE_COLUMNS


class SecurityEnv(gym.Env):
    """Walk a preprocessed interactions DataFrame, one row per step."""

    def __init__(self, data):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.action_space = spaces.Discrete(3)  # [0=isolate, 1=scan, 2=alert]
        # The observation is the two features built in Step One, unbounded:
        # query_encoded is an arbitrary integer code and the normalized
        # feedback can be negative.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.current_step = 0

    def _observation(self):
        row = self.data.iloc[self.current_step]
        return row[FEATURE_COLUMNS].values.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._observation(), {}

    def step(self, action):
        row = self.data.iloc[self.current_step]
        reward = 1.0 if row["threat_resolved"] else -1.0
        reward += float(row["feedback_normalized"])
        self.current_step += 1
        done = self.current_step >= len(self.data)
        if done:
            # Terminal observation: repeat the last row's features. The
            # original script indexed one past the end here and, in
            # practice, the training loop broke out on `done` before the
            # value was ever used as a real state.
            next_obs = self.data.iloc[-1][FEATURE_COLUMNS].values.astype(np.float32)
        else:
            next_obs = self._observation()
        return next_obs, reward, done, False, {}
