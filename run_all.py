"""Single entry point: run the whole threat-response pipeline end-to-end.

    python run_all.py [--episodes N] [--data PATH] [--model-out PATH]

Equivalent to running pipeline.py directly; it exists so the repo has one
obvious command that does everything (data prep -> RL training -> A/B
evaluation -> saved model bundle for app.py).
"""
from pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
