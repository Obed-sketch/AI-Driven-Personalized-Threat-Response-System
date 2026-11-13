"""End-to-end pipeline (Steps One-Three + the A/B evaluation of Step Four).

Run directly or via run_all.py:

    python pipeline.py --episodes 200

Stages: load + preprocess -> train the Q-network on the training split ->
A/B evaluation on the held-out test split (RL policy vs the rule-based
'always scan' baseline, same reward the environment gives) -> save the
serving bundle (weights + scaler + query mapping) that app.py loads.
"""
import argparse
import time
from pathlib import Path

from threat_response.agent import RLAgent, evaluate_policy, rule_based_policy, train_agent
from threat_response.env import SecurityEnv
from threat_response.model_io import save_bundle
from threat_response.preprocessing import load_interactions, preprocess, split

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "user_interactions.csv"
DEFAULT_MODEL = BASE_DIR / "rl_agent_bundle.pt"


def run_pipeline(data_path=DEFAULT_DATA, episodes=200, seed=42,
                 model_path=DEFAULT_MODEL, log_mlflow=True):
    start = time.time()
    raw = load_interactions(data_path)
    data, scaler, query_to_code = preprocess(raw)
    train_data, test_data = split(data)

    train_env = SecurityEnv(train_data)
    network, episode_rewards = train_agent(train_env, episodes=episodes,
                                           seed=seed)

    # A/B evaluation on the held-out split: identical passes over the test
    # rows, one per policy, scored with the environment's reward.
    rl_agent = RLAgent(network)
    rl_total, rl_mean = evaluate_policy(
        SecurityEnv(test_data), rl_agent.predict)
    rule_total, rule_mean = evaluate_policy(
        SecurityEnv(test_data), rule_based_policy)

    save_bundle(model_path, network, scaler, query_to_code)

    summary = {
        "rows_total": int(len(data)),
        "rows_train": int(len(train_data)),
        "rows_test": int(len(test_data)),
        "episodes": int(episodes),
        "final_episode_reward": float(episode_rewards[-1]),
        "mean_episode_reward_last_10": float(
            sum(episode_rewards[-10:]) / len(episode_rewards[-10:])),
        "ab_test_rl_total_reward": float(rl_total),
        "ab_test_rl_mean_reward": float(rl_mean),
        "ab_test_rule_total_reward": float(rule_total),
        "ab_test_rule_mean_reward": float(rule_mean),
        "model_path": str(model_path),
        "elapsed_seconds": time.time() - start,
    }

    if log_mlflow:
        try:
            import mlflow

            with mlflow.start_run(run_name="run_all"):
                mlflow.log_params({
                    "episodes": episodes,
                    "seed": seed,
                    "rows_train": summary["rows_train"],
                    "rows_test": summary["rows_test"],
                })
                mlflow.log_metrics({
                    "final_episode_reward": summary["final_episode_reward"],
                    "ab_rl_mean_reward": rl_mean,
                    "ab_rule_mean_reward": rule_mean,
                })
        except Exception as exc:  # tracking must never sink the pipeline
            print(f"note: MLflow logging skipped ({exc})")

    return summary


def format_summary(summary):
    return "\n".join([
        "=== Threat-response pipeline summary ===",
        f"Data: {summary['rows_total']} rows "
        f"({summary['rows_train']} train / {summary['rows_test']} test)",
        f"Training: {summary['episodes']} episodes, "
        f"final-episode reward {summary['final_episode_reward']:.2f}, "
        f"mean of last 10 episodes "
        f"{summary['mean_episode_reward_last_10']:.2f}",
        "A/B evaluation on the held-out test split (environment reward):",
        f"  RL policy        total {summary['ab_test_rl_total_reward']:.2f} "
        f"(mean {summary['ab_test_rl_mean_reward']:.3f}/step)",
        f"  Rule-based base. total "
        f"{summary['ab_test_rule_total_reward']:.2f} "
        f"(mean {summary['ab_test_rule_mean_reward']:.3f}/step)",
        f"Model bundle saved to {summary['model_path']}",
        f"Elapsed: {summary['elapsed_seconds']:.1f}s",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="interactions CSV (default: user_interactions.csv)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="training episodes (default: 200; the original "
                             "script used 1000, which takes several minutes "
                             "on CPU with the sample data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL),
                        help="where to save the serving bundle")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="skip MLflow metric logging")
    args = parser.parse_args(argv)

    summary = run_pipeline(
        data_path=args.data,
        episodes=args.episodes,
        seed=args.seed,
        model_path=args.model_out,
        log_mlflow=not args.no_mlflow,
    )
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
