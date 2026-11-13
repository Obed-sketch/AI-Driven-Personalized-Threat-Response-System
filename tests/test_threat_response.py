"""Tests for the threat-response pipeline, environment, agent and API.

Run from the repo root:  python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import run_pipeline  # noqa: E402
from threat_response.agent import (  # noqa: E402
    RLAgent,
    evaluate_policy,
    rule_based_policy,
    train_agent,
)
from threat_response.env import SecurityEnv  # noqa: E402
from threat_response.model_io import load_bundle, save_bundle  # noqa: E402
from threat_response.preprocessing import (  # noqa: E402
    load_interactions,
    preprocess,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = REPO_ROOT / "user_interactions.csv"


@pytest.fixture(scope="module")
def processed():
    raw = load_interactions(SAMPLE_CSV)
    return preprocess(raw)  # (data, scaler, query_to_code)


def test_preprocessing_adds_features(processed):
    data, scaler, query_to_code = processed
    assert len(data) == 120
    assert set(data["query_encoded"].unique()) == set(range(6))
    # StandardScaler output: ~zero mean, unit variance
    assert abs(data["feedback_normalized"].mean()) < 1e-6
    assert abs(data["feedback_normalized"].std(ddof=0) - 1.0) < 1e-6
    assert query_to_code.keys() == set(data["query"].unique())


def test_env_reset_and_step(processed):
    data, _, _ = processed
    env = SecurityEnv(data.head(5))
    obs, info = env.reset()
    assert obs.shape == (2,)
    steps = 0
    done = False
    while not done:
        obs, reward, done, truncated, _ = env.step(1)
        assert isinstance(reward, float)
        steps += 1
    assert steps == 5


def test_env_reward_matches_first_row(processed):
    data, _, _ = processed
    env = SecurityEnv(data.head(3))
    env.reset()
    _, reward, *_ = env.step(0)
    first = data.iloc[0]
    expected = (1.0 if first["threat_resolved"] else -1.0) \
        + float(first["feedback_normalized"])
    assert reward == pytest.approx(expected)


def test_rule_based_policy_always_scans():
    assert rule_based_policy(np.zeros(2, dtype=np.float32)) == 1


def test_agent_actions_are_valid(processed):
    data, scaler, query_to_code = processed
    env = SecurityEnv(data.head(20))
    network, rewards = train_agent(env, episodes=3, seed=1)
    assert len(rewards) == 3
    agent = RLAgent(network)
    for obs in ([0.0, 0.0], [3.0, 1.5], [5.0, -2.0]):
        assert agent.predict(np.array(obs, dtype=np.float32)) in (0, 1, 2)


def test_evaluate_policy_runs(processed):
    data, _, _ = processed
    total, mean = evaluate_policy(SecurityEnv(data.head(10)),
                                  rule_based_policy)
    assert total == pytest.approx(mean * 10)


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("model") / "bundle.pt"
    run_pipeline(data_path=SAMPLE_CSV, episodes=3, seed=3,
                 model_path=out, log_mlflow=False)
    return out


def test_bundle_roundtrip(bundle_path, processed):
    _, scaler, query_to_code = processed
    assert bundle_path.exists()
    agent, meta = load_bundle(bundle_path)
    assert meta["query_to_code"] == query_to_code
    assert meta["action_names"] == ["isolate", "scan", "alert"]
    assert agent.predict(np.array([1.0, 0.5], dtype=np.float32)) in (0, 1, 2)


@pytest.fixture(scope="module")
def client(bundle_path):
    from app import create_app

    return create_app(bundle_path).test_client()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok", "model_loaded": True}


def test_recommend_endpoint_raw_context(client):
    resp = client.post("/recommend?policy=rl",
                       json={"query": "malware alert", "feedback": 4})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["policy"] == "rl"
    assert body["action"] in (0, 1, 2)
    assert body["action_name"] in ("isolate", "scan", "alert")


def test_recommend_endpoint_rule_policy(client):
    resp = client.post("/recommend?policy=rule",
                       json={"query_encoded": 2, "feedback": 3})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["policy"] == "rule_based"
    assert body["action"] == 1  # baseline always scans
    assert body["action_name"] == "scan"


def test_recommend_ab_split_returns_valid_actions(client):
    for _ in range(10):
        resp = client.post("/recommend",
                           json={"query": "phishing email", "feedback": 2})
        assert resp.status_code == 200
        assert resp.get_json()["action"] in (0, 1, 2)


def test_recommend_rejects_empty_payload(client):
    resp = client.post("/recommend", json={})
    assert resp.status_code == 400
