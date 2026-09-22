#!/usr/bin/env python
"""Reproduce selected scMDCL/scDPCL experiments and collect their metrics."""

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("datasets.json")
ALL_DATASETS = ("PBMC-10k", "PBMC-3k", "BMNC")

UNIFIED_ARGS = [
    "--seed", "0",
    "--pretrain",
    "--train_after_pretrain",
    "--k", "10",
    "--n_z", "20",
    "--disentangle_mode", "hard",
    "--n_shared", "16",
    "--n_private", "4",
    "--alpha1", "0.1",
    "--alpha2", "10",
    "--target_power", "2",
    "--beta", "0",
    "--gamma", "0",
    "--pretrain_gamma", "0",
    "--lambda_cell", "1",
    "--cell_loss", "legacy",
    "--warmup_epochs", "100",
    "--beta_ramp_epochs", "100",
    "--gamma_start_epochs", "200",
    "--gamma_ramp_epochs", "100",
    "--dropout", "0.4",
    "--center_mode", "train",
    "--eval_q_mode", "adaptive",
    "--eval_view_weight", "0.65",
    "--eval_firnd_weight", "0.6",
    "--eval_q_power", "1",
    "--eval_ema_decay", "0.95",
    "--cluster_recovery_splits", "2",
    "--cluster_recovery_pca", "5",
    "--cluster_recovery_view1_weight", "3",
]

DATASET_ARGS = {
    "PBMC-10k": ["--n_d1", "100", "--n_d2", "100", "--lr", "0.0005"],
    "PBMC-3k": ["--n_d1", "100", "--n_d2", "49", "--lr", "0.0008"],
    "BMNC": ["--n_d1", "100", "--n_d2", "25", "--lr", "0.001"],
}

LEGACY_TUNED_ARGS = {
    "PBMC-10k": [
        "--seed", "0",
        "--scmdcl_init",
        "--k", "20",
        "--multi_k", "15,20,25",
        "--n_d1", "100",
        "--n_d2", "100",
        "--disentangle_mode", "hard",
        "--n_shared", "20",
        "--n_private", "0",
        "--alpha2", "7.5",
        "--target_power", "2",
        "--lr", "0.0008",
        "--beta", "0",
        "--gamma", "0",
        "--lambda_cell", "1",
        "--cell_loss", "legacy",
        "--dropout", "0",
        "--center_mode", "eval",
        "--eval_q_mode", "adaptive",
        "--eval_view_weight", "0.35",
        "--eval_firnd_weight", "0.4",
        "--eval_q_power", "1",
        "--eval_ema_decay", "0.975",
        "--cluster_recovery_splits", "2",
        "--cluster_recovery_pca", "20",
        "--cluster_recovery_view1_weight", "1.25",
    ],
    "PBMC-3k": [
        "--seed", "0",
        "--pretrain",
        "--train_after_pretrain",
        "--k", "10",
        "--n_d1", "100",
        "--n_d2", "49",
        "--disentangle_mode", "hard",
        "--n_shared", "16",
        "--n_private", "4",
        "--alpha2", "10",
        "--target_power", "2",
        "--lr", "0.0008",
        "--beta", "0",
        "--gamma", "0",
        "--lambda_cell", "1",
        "--cell_loss", "legacy",
        "--dropout", "0.4",
        "--center_mode", "train",
        "--eval_q_mode", "adaptive",
        "--eval_view_weight", "0.75",
        "--eval_firnd_weight", "0.6",
        "--eval_ema_decay", "0.95",
        "--cluster_recovery_splits", "1",
        "--cluster_recovery_pca", "5",
        "--cluster_recovery_view1_weight", "3",
    ],
    "BMNC": [
        "--seed", "0",
        "--pretrain",
        "--train_after_pretrain",
        "--k", "10",
        "--n_d1", "100",
        "--n_d2", "25",
        "--disentangle_mode", "hard",
        "--n_shared", "16",
        "--n_private", "4",
        "--alpha2", "10",
        "--target_power", "2",
        "--lr", "0.001",
        "--beta", "0",
        "--gamma", "0",
        "--lambda_cell", "1",
        "--cell_loss", "legacy",
        "--dropout", "0.4",
        "--center_mode", "eval",
        "--eval_q_mode", "firnd",
        "--eval_view_weight", "0.65",
    ],
}

TWO_GROUP_ARGS = {
    "PBMC-10k": LEGACY_TUNED_ARGS["PBMC-10k"],
    "PBMC-3k": [*UNIFIED_ARGS, *DATASET_ARGS["PBMC-3k"]],
    "BMNC": [*UNIFIED_ARGS, *DATASET_ARGS["BMNC"]],
}

METRIC_PATTERN = (
    r"ARI:\s*(?P<ari>[0-9.]+),\s*NMI:\s*(?P<nmi>[0-9.]+),\s*"
    r"AMI:\s*(?P<ami>[0-9.]+),?\s*ACC:\s*(?P<acc>[0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PBMC-10k, PBMC-3k, and BMNC reproduction experiments."
    )
    parser.add_argument(
        "--datasets",
        default=",".join(ALL_DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--mode",
        choices=("optimized", "original"),
        default="optimized",
        help="Run the optimized scDPCL configuration or original scMDCL baseline.",
    )
    parser.add_argument(
        "--profile",
        choices=("unified", "two_group", "legacy_tuned"),
        default="unified",
        help=(
            "Use one shared configuration, a PBMC-10k/PBMC-3k+BMNC "
            "two-group protocol, or historical per-dataset settings."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the target conda environment.",
    )
    parser.add_argument(
        "--output_root",
        default="",
        help="Log directory. A timestamped directory is used by default.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and print commands without starting training.",
    )
    parser.add_argument(
        "--reuse_original_pretrain",
        action="store_true",
        help="In original mode, skip pretraining when the checkpoint already exists.",
    )
    return parser.parse_args()


def load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_datasets(raw_names):
    names = [name.strip() for name in raw_names.split(",") if name.strip()]
    unknown = sorted(set(names) - set(ALL_DATASETS))
    if unknown:
        raise ValueError("Unknown dataset(s): {}".format(", ".join(unknown)))
    if not names:
        raise ValueError("At least one dataset is required.")
    return names


def required_files(name, info, mode, profile):
    data_dir = ROOT / "input" / name
    files = [
        data_dir / "RNA_fea.npy",
        data_dir / "{}_fea.npy".format(info["second_view"]),
        data_dir / "label.npy",
    ]
    graph_k = [20]
    if mode == "optimized" and profile == "unified":
        graph_k = [10]
    elif mode == "optimized" and profile == "two_group":
        graph_k = info["available_k"] if name == "PBMC-10k" else [10]
    elif mode == "optimized" and profile == "legacy_tuned":
        graph_k = info["available_k"] if name == "PBMC-10k" else [10]
    for k_value in graph_k:
        files.extend(
            [
                data_dir / "RNA_euc_{}.npz".format(k_value),
                data_dir / "{}_euc_{}.npz".format(info["second_view"], k_value),
            ]
        )
    if (
        mode == "optimized"
        and profile in ("two_group", "legacy_tuned")
        and name == "PBMC-10k"
    ):
        files.append(ROOT / "model_pretrained" / "{}_pretrain.pkl".format(name))
    return files


def validate_inputs(datasets, manifest, mode, profile):
    missing = []
    for name in datasets:
        missing.extend(
            path
            for path in required_files(name, manifest[name], mode, profile)
            if not path.is_file()
        )
    if missing:
        formatted = "\n".join("  - {}".format(path) for path in missing)
        raise FileNotFoundError("Required reproduction files are missing:\n{}".format(formatted))


def optimized_command(name, python_executable, output_root, profile):
    output_dir = output_root / "outputs" / name
    if profile == "unified":
        model_args = [*UNIFIED_ARGS, *DATASET_ARGS[name]]
    elif profile == "two_group":
        model_args = TWO_GROUP_ARGS[name]
    else:
        model_args = LEGACY_TUNED_ARGS[name]
    return [
        python_executable,
        str(ROOT / "model" / "main_dpcl.py"),
        "--name",
        name,
        *model_args,
        "--output_dir",
        str(output_dir),
    ]


def original_commands(name, info, python_executable, reuse_pretrain):
    common = [
        "--name", name,
        "--k", "20",
        "--n_d1", str(info["rna_dim"]),
        "--n_d2", str(info["second_dim"]),
        "--second_view", info["second_view"],
    ]
    checkpoint = ROOT / "model_pretrained" / "{}_pretrain.pkl".format(name)
    commands = []
    if not (reuse_pretrain and checkpoint.is_file()):
        commands.append(("pretrain", [python_executable, str(ROOT / "main.py"), *common, "--pretrain", "True"]))
    commands.append(("train", [python_executable, str(ROOT / "main.py"), *common]))
    return commands


def command_text(command):
    return subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)


def run_command(command, log_path):
    print("$ {}".format(command_text(command)))
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            log_handle.write(line)
            if "Best_epoch:" in line or "Recovered:" in line or "Final_epoch:" in line:
                print(line.rstrip())
        return process.wait()


def read_log_text(log_path):
    raw = log_path.read_bytes()
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "gb18030")
    candidates = []
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        score = sum(text.count(marker) for marker in ("Best_epoch:", "Final_epoch:", "Recovered:"))
        candidates.append((score, -text.count("\x00"), text))
    if not candidates:
        return raw.decode("utf-8", errors="replace")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def parse_metrics(log_path):
    text = read_log_text(log_path)
    result = {}
    prefixes = {
        "best": r"Best_epoch:\s*(?P<epoch>\d+),\s*",
        "final": r"Final_epoch:\s*(?P<epoch>\d+),\s*",
        "recovered": r"(?<!Final_)Recovered:\s*",
        "final_recovered": r"Final_recovered:\s*",
        "rare_recovered": r"Rare_recovered:\s*",
    }
    for key, prefix in prefixes.items():
        matches = list(re.finditer(prefix + METRIC_PATTERN, text, flags=re.IGNORECASE))
        if not matches:
            continue
        values = matches[-1].groupdict()
        for metric in ("ari", "nmi", "ami", "acc"):
            result["{}_{}".format(key, metric)] = float(values[metric])
        if values.get("epoch") is not None:
            result["{}_epoch".format(key)] = int(values["epoch"])
    return result


def write_summary(rows, output_root):
    json_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def main():
    args = parse_args()
    manifest = load_manifest()
    datasets = select_datasets(args.datasets)
    validate_inputs(datasets, manifest, args.mode, args.profile)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else ROOT / "model" / "experiment_logs" / "reproduction_selected_{}".format(timestamp)
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    print("Mode: {}".format(args.mode))
    print("Profile: {}".format(args.profile))
    print("Datasets: {}".format(", ".join(datasets)))
    print("Output: {}".format(output_root))

    planned = []
    for name in datasets:
        if args.mode == "optimized":
            stage = "optimized_{}".format(args.profile)
            planned.append(
                (
                    name,
                    stage,
                    optimized_command(name, args.python, output_root, args.profile),
                )
            )
        else:
            for stage, command in original_commands(
                name, manifest[name], args.python, args.reuse_original_pretrain
            ):
                planned.append((name, stage, command))

    if args.dry_run:
        for name, stage, command in planned:
            print("[{}:{}] {}".format(name, stage, command_text(command)))
        print("Dry run passed: all required input files are present.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "commands.txt").write_text(
        "\n".join(
            "[{}:{}] {}".format(name, stage, command_text(command))
            for name, stage, command in planned
        )
        + "\n",
        encoding="utf-8",
    )

    rows = []
    for name, stage, command in planned:
        log_path = output_root / "{}_{}.log".format(name, stage)
        returncode = run_command(command, log_path)
        row = {
            "dataset": name,
            "mode": args.mode,
            "stage": stage,
            "returncode": returncode,
            "log": str(log_path),
            "command": command_text(command),
        }
        row.update(parse_metrics(log_path))
        rows.append(row)
        write_summary(rows, output_root)
        if returncode != 0:
            print("Experiment failed; see {}".format(log_path), file=sys.stderr)
            return returncode

    csv_path, json_path = write_summary(rows, output_root)
    print("Completed. Summary: {} and {}".format(csv_path, json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
