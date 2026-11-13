"""Threat-response RL package.

Modules extracted from the original single-file script
("AI_Driven Personalized Threat Response System.py") so the pipeline can be
run end-to-end (run_all.py), served (app.py) and unit-tested (tests/).
The logic is the same as in the original script: two features per
observation (query_encoded, feedback_normalized), three actions
(0=isolate, 1=scan, 2=alert), reward = +1/-1 for threat resolved plus the
normalized feedback score.
"""

ACTION_NAMES = ["isolate", "scan", "alert"]
