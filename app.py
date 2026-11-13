"""Step Five, implemented: serve the trained threat-response policy.

    python run_all.py          # first: train and save rl_agent_bundle.pt
    python app.py              # dev server on http://127.0.0.1:5000
    gunicorn --bind 0.0.0.0:5000 app:app   # production-style (see Dockerfile)

Endpoints:
    GET  /health     -> {"status": "ok", "model_loaded": true}
    POST /recommend  -> A/B-served recommendation (Step Four's design):
                        half the requests go to the RL policy, half to the
                        rule-based baseline, unless ?policy=rl or
                        ?policy=rule pins the choice.
    GET  /metrics    -> Prometheus counters (rl_recommendations,
                        rule_recommendations), as in the original design.

/recommend accepts any of:
    {"query": "malware alert", "feedback": 4}          (raw, normalized here)
    {"query_encoded": 2, "feedback": 4}
    {"query_encoded": 2, "feedback_normalized": 0.35}  (pre-encoded)
and returns {"action": int, "action_name": str, "policy": "rl"|"rule_based"}.
Valid actions: 0=isolate, 1=scan, 2=alert.
"""
import os
import random
from pathlib import Path

import prometheus_client as prom
from flask import Flask, Response, request

from threat_response.model_io import encode_context, load_bundle

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "rl_agent_bundle.pt"

counter_rl = prom.Counter("rl_recommendations", "RL model recommendations")
counter_rule = prom.Counter(
    "rule_recommendations", "Rule-based recommendations")


def create_app(model_path=None):
    model_path = Path(model_path or os.environ.get("MODEL_PATH",
                                                    DEFAULT_MODEL))
    if not model_path.exists():
        raise FileNotFoundError(
            f"model bundle not found at {model_path}; "
            "run `python run_all.py` first to train and save it"
        )
    agent, meta = load_bundle(model_path)
    action_names = meta["action_names"]

    app = Flask(__name__)
    app.config["MODEL_META"] = meta

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True}

    @app.post("/recommend")
    def recommend():
        payload = request.get_json(silent=True) or {}
        try:
            observation = encode_context(meta, **{
                key: payload[key]
                for key in ("query", "feedback", "query_encoded",
                            "feedback_normalized")
                if key in payload
            })
        except (ValueError, TypeError) as exc:
            return {"error": str(exc)}, 400

        policy = request.args.get("policy")
        if policy not in ("rl", "rule"):
            policy = "rl" if random.random() < 0.5 else "rule"

        if policy == "rl":
            action = agent.predict(observation)
            counter_rl.inc()
            policy_name = "rl"
        else:
            from threat_response.agent import rule_based_policy

            action = rule_based_policy(observation)
            counter_rule.inc()
            policy_name = "rule_based"
        return {
            "action": int(action),
            "action_name": action_names[int(action)],
            "policy": policy_name,
        }

    @app.get("/metrics")
    def metrics():
        return Response(prom.generate_latest(),
                        mimetype="text/plain; version=0.0.4")

    return app


# Gunicorn entry point (gunicorn app:app). Creating it at import time means
# the bundle must exist -- in the Docker image it is trained during the
# image build; locally, run `python run_all.py` first.
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
