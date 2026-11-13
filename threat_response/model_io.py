"""Save/load the trained policy as a single bundle file.

The original script saved only the network weights (rl_agent.pth). Serving
also needs the preprocessing state (the feedback scaler and the
query -> code mapping), so the bundle stores all of it in one torch file.
"""
import numpy as np
import torch

from . import ACTION_NAMES
from .agent import QNetwork, RLAgent


def save_bundle(path, network, scaler, query_to_code):
    bundle = {
        "state_dict": network.state_dict(),
        "input_dim": int(network.fc1.in_features),
        "output_dim": int(network.fc2.out_features),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "query_to_code": {str(k): int(v) for k, v in query_to_code.items()},
        "action_names": list(ACTION_NAMES),
    }
    torch.save(bundle, path)
    return path


def load_bundle(path):
    """Returns (agent, meta): an RLAgent ready to predict, and a dict with
    the scaler statistics, query mapping and action names."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    network = QNetwork(bundle["input_dim"], bundle["output_dim"])
    network.load_state_dict(bundle["state_dict"])
    network.eval()
    meta = {
        "scaler_mean": float(np.asarray(bundle["scaler_mean"]).ravel()[0]),
        "scaler_scale": float(np.asarray(bundle["scaler_scale"]).ravel()[0]),
        "query_to_code": bundle["query_to_code"],
        "action_names": bundle["action_names"],
    }
    return RLAgent(network), meta


def encode_context(meta, query=None, feedback=None,
                   query_encoded=None, feedback_normalized=None):
    """Build the 2-feature observation from a request context.

    Accepts either raw values (query text + 1-5 feedback, normalized with
    the training scaler) or pre-encoded features. A query string never seen
    in training maps to a new code one past the training maximum, mirroring
    how an unseen category would have to be handled."""
    if query_encoded is None:
        if query is None:
            raise ValueError("provide either 'query' or 'query_encoded'")
        mapping = meta["query_to_code"]
        if query in mapping:
            query_encoded = mapping[query]
        else:
            query_encoded = (max(mapping.values()) + 1) if mapping else 0
    if feedback_normalized is None:
        if feedback is None:
            raise ValueError(
                "provide either 'feedback' or 'feedback_normalized'")
        feedback_normalized = (
            float(feedback) - meta["scaler_mean"]
        ) / meta["scaler_scale"]
    return np.array(
        [float(query_encoded), float(feedback_normalized)], dtype=np.float32
    )
