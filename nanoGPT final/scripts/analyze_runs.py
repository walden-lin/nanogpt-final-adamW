#!/usr/bin/env python
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_logs(root):
    runs = {}
    for path in glob.glob(os.path.join(root, "*", "logs.txt")):
        run_name = Path(path).parent.name
        entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if entries:
            runs[run_name] = entries
    return runs


def extract_series(entries, split, key):
    xs, ys = [], []
    for e in entries:
        if e.get("split") != split:
            continue
        if key in e:
            xs.append(e.get("iter", 0))
            ys.append(e[key])
        elif split == "val" and key == "val_loss" and "val_loss" in e:
            xs.append(e.get("iter", 0))
            ys.append(e["val_loss"])
    return xs, ys


def plot_metric(runs, run_names, split, key, title, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    for name in run_names:
        entries = runs.get(name, [])
        xs, ys = extract_series(entries, split, key)
        if xs:
            plt.plot(xs, ys, label=f"{name}-{split}")
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_lr(runs, run_names, out_path):
    plt.figure(figsize=(8, 5))
    for name in run_names:
        entries = runs.get(name, [])
        xs, ys = extract_series(entries, "train", "lr")
        if xs:
            plt.plot(xs, ys, label=name)
    plt.title("Learning rate vs iteration")
    plt.xlabel("iteration")
    plt.ylabel("learning rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def pick_default_runs(runs):
    tags = [
        "adamw_constant",
        "adamw_cosine_wr",
        "adamw_cosine",
        "adaml2_constant",
        "adaml2_cosine",
    ]
    picked = []
    for name in runs:
        for tag in tags:
            if tag in name and name not in picked:
                picked.append(name)
    return picked or list(runs.keys())[:5]


def extract_config(entries):
    """Grab representative hyperparameters from the latest log entry."""
    if not entries:
        return {}
    e = entries[-1]
    keys = [
        "optim_mode",
        "lr_schedule",
        "warm_restarts",
        "learning_rate",
        "weight_decay",
        "l2_lambda",
        "l2_target",
    ]
    cfg = {k: e.get(k) for k in keys if k in e}
    # lr stored as 'lr' in logs
    if "lr" in e:
        cfg["lr"] = e["lr"]
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Analyze optimizer experiment logs.")
    parser.add_argument("--root", default="out", help="Root directory containing run subfolders.")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Specific run directory names to plot (default: auto-pick a representative subset).",
    )
    parser.add_argument("--fig_dir", default="figures", help="Output directory for plots.")
    args = parser.parse_args()

    runs = load_logs(args.root)
    if not runs:
        raise SystemExit(f"No logs found under {args.root}")

    run_names = args.runs if args.runs else pick_default_runs(runs)
    os.makedirs(args.fig_dir, exist_ok=True)

    plot_metric(runs, run_names, "train", "loss", "Training Loss", "loss", os.path.join(args.fig_dir, "loss_train.png"))
    plot_metric(runs, run_names, "val", "val_loss", "Validation Loss", "loss", os.path.join(args.fig_dir, "loss_val.png"))
    plot_metric(
        runs, run_names, "train", "param_norm", "Parameter L2 Norm", "||theta||_2", os.path.join(args.fig_dir, "param_norm.png")
    )
    plot_metric(
        runs, run_names, "train", "grad_norm", "Gradient L2 Norm", "||grad||_2", os.path.join(args.fig_dir, "grad_norm.png")
    )
    plot_lr(runs, run_names, os.path.join(args.fig_dir, "lr_schedules.png"))

    summary_path = os.path.join(args.fig_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Optimizer comparison summary\n\n")
        f.write("Plotted runs:\n")
        for name in run_names:
            f.write(f"- {name}\n")
        f.write("\nRun hyperparameters (from logs):\n\n")
        f.write("| run | optim_mode | lr_schedule | warm_restarts | lr | weight_decay | l2_lambda | l2_target |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for name in run_names:
            cfg = extract_config(runs.get(name, []))
            f.write(
                f"| {name} | {cfg.get('optim_mode','')} | {cfg.get('lr_schedule','')} | {cfg.get('warm_restarts','')} | "
                f"{cfg.get('lr', cfg.get('learning_rate',''))} | {cfg.get('weight_decay','')} | "
                f"{cfg.get('l2_lambda','')} | {cfg.get('l2_target','')} |\n"
            )
        f.write("\nFigures:\n")
        for fname in ["loss_train.png", "loss_val.png", "param_norm.png", "grad_norm.png", "lr_schedules.png"]:
            f.write(f"- {os.path.join(args.fig_dir, fname)}\n")
    print(f"Saved figures and summary to {args.fig_dir}")


if __name__ == "__main__":
    main()
