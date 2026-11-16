"""Step Two of the original script: the custom Gym environment.

The environment walks the dataset one row at a time; each row is one alert
context (query type + user feedback). The gym>=0.26 API is used: reset()
returns (observation, info) and step() returns a 5-tuple.

REWARD -- v2 (Nov 2026). What was wrong in v1: the reward was
"+1 if the row's threat_resolved flag was set, -1 otherwise, plus the
row's normalized feedback". None of that depends on the action the agent
takes -- threat_resolved is a historical outcome recorded in the data,
and the feedback was given before the agent acted. An agent therefore
scored exactly the same whether it isolated a ransomware host or merely
sent a notification, and the A/B test tied (RL 9.47 vs rule-based 9.47):
there was literally nothing to learn.

The v2 reward grades the *decision* instead. For each alert type there is
a severity-appropriate first response (standard incident-response
practice: contain active host compromise, investigate anomalies, notify
on identity events):

    ransomware warning        -> isolate   (critical: containment first)
    malware alert             -> isolate   (high: contain the host)
    unusual outbound traffic  -> scan      (medium: investigate / scope)
    phishing email            -> scan      (medium: scan to find scope)
    suspicious login          -> alert     (low: notify user / admin)
    password reset request    -> alert     (low: notify / verify)

Actions also have a response strength: isolate (2) > scan (1) > alert (0).
The reward's main term compares the chosen action with the appropriate
one:

    appropriate action          +2.0
    under-response (weaker)     -1.5   (a severe threat gets a memo)
    over-response (stronger)    -0.5   (disruptive, but safe)

Under-responding is penalized more than over-responding on purpose: in
security operations, failing to contain a real threat costs far more
than isolating a machine that turns out to be clean.

The row's normalized feedback then *modulates* the reward
(+0.5 * feedback_normalized): user satisfaction matters, but the term is
identical for every action on the same row, so it can never make a wrong
response score better than the right one -- the appropriateness gap
(at least 2.5) always dominates it. The historical threat_resolved flag
is no longer part of the reward at all: the agent cannot influence a
recorded past outcome, so paying it for one taught it nothing.
"""
import gym
import numpy as np
from gym import spaces

from .preprocessing import FEATURE_COLUMNS

# Severity-appropriate first response per alert type (see module docstring).
APPROPRIATE_ACTION = {
    "ransomware warning": 0,        # isolate
    "malware alert": 0,             # isolate
    "unusual outbound traffic": 1,  # scan
    "phishing email": 1,            # scan
    "suspicious login": 2,          # alert
    "password reset request": 2,    # alert
}
# Unknown alert types default to "scan" (investigate first) -- the safe
# middle response when severity has not been classified.
DEFAULT_APPROPRIATE_ACTION = 1

# Response strength: isolate is the strongest intervention, alert the
# lightest. Used to distinguish under- from over-responding.
ACTION_STRENGTH = {0: 2, 1: 1, 2: 0}

REWARD_APPROPRIATE = 2.0
REWARD_UNDER_RESPONSE = -1.5
REWARD_OVER_RESPONSE = -0.5
FEEDBACK_WEIGHT = 0.5


def appropriate_action(query):
    """The severity-appropriate first response for an alert type."""
    return APPROPRIATE_ACTION.get(query, DEFAULT_APPROPRIATE_ACTION)


def response_reward(query, action, feedback_normalized):
    """v2 reward for taking `action` on an alert of type `query`.

    Action-dependent by construction: for the same context, the
    appropriate response scores +2.0 (plus feedback modulation), an
    under-response -1.5, an over-response -0.5.
    """
    needed = appropriate_action(query)
    if int(action) == needed:
        match = REWARD_APPROPRIATE
    elif ACTION_STRENGTH[int(action)] < ACTION_STRENGTH[needed]:
        match = REWARD_UNDER_RESPONSE
    else:
        match = REWARD_OVER_RESPONSE
    return match + FEEDBACK_WEIGHT * float(feedback_normalized)


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
        reward = response_reward(
            row["query"], action, row["feedback_normalized"])
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
