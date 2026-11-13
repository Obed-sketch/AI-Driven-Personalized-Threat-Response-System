"""Generate a small SYNTHETIC sample of user_interactions.csv.

The real interaction data used with this project was never committed to the
repo. This generator creates 120 clearly synthetic rows with the schema the
main script expects (user_id, query, threat_resolved, feedback, timestamp)
so the Step One-Three pipeline can be run and verified. The feedback values
are random; no result produced from this file demonstrates anything about
real users.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
queries = [
    "malware alert",
    "phishing email",
    "suspicious login",
    "ransomware warning",
    "unusual outbound traffic",
    "password reset request",
]

n = 120
data = pd.DataFrame({
    "user_id": rng.integers(1, 21, size=n),
    "query": rng.choice(queries, size=n),
    "threat_resolved": rng.random(n) < 0.7,
    "feedback": rng.integers(1, 6, size=n),  # 1-5 scale, as in the README
    "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
})
data.to_csv("user_interactions.csv", index=False)
print(f"Wrote user_interactions.csv ({len(data)} synthetic rows)")
