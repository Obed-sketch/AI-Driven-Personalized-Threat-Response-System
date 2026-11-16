"""Threat-response RL package.

Modules extracted from the original single-file script
("AI_Driven Personalized Threat Response System.py") so the pipeline can be
run end-to-end (run_all.py), served (app.py) and unit-tested (tests/).
Two features per observation (query_encoded, feedback_normalized), three
actions (0=isolate, 1=scan, 2=alert). The reward is the v2 design in
env.py: it grades the chosen action against the severity-appropriate
response for the alert type (+2.0 appropriate, -1.5 under-response,
-0.5 over-response), modulated by the normalized feedback score. (The
original script's reward -- +1/-1 for the row's recorded threat_resolved
flag plus feedback -- did not depend on the action at all; see env.py.)
"""

ACTION_NAMES = ["isolate", "scan", "alert"]
