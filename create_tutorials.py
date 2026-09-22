"""Generate the three reproducible experiment notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TUTORIAL_DIR = ROOT / "tutorial"

DATASETS = {
    "PBMC-10k": {
        "file": "tutorial_pbmc10k.ipynb",
        "description": "PBMC 10k RNA-ATAC 多模态单细胞数据",
    },
    "PBMC-3k": {
        "file": "tutorial_pbmc3k.ipynb",
        "description": "PBMC 3k RNA-ATAC 多模态单细胞数据",
    },
    "BMNC": {
        "file": "tutorial_bmnc.ipynb",
        "description": "BMNC RNA-ATAC 多模态单细胞数据",
    },
}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(dataset, description):
    cells = [
        markdown(
            f"# {dataset} 复现实验\n\n"
            f"{description}。本 notebook 默认使用 `two_group` 协议，并将结果写入发布包的 "
            f"`outputs/{dataset}/`。"
        ),
        code(
            "import json\n"
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "def find_project_root(start):\n"
            "    for path in (start, *start.parents):\n"
            "        if (path / 'model' / 'main_dpcl.py').is_file() and (path / 'scDPCL_release').is_dir():\n"
            "            return path\n"
            "    raise FileNotFoundError('无法定位 scMDCL-main 项目根目录')\n\n"
            "PROJECT_ROOT = find_project_root(Path.cwd().resolve())\n"
            "RELEASE_ROOT = PROJECT_ROOT / 'scDPCL_release'\n"
            "os.chdir(PROJECT_ROOT)\n"
            f"DATASET = {dataset!r}\n"
            "PROFILE = 'two_group'\n"
            "print('Python:', sys.executable)\n"
            "print('Project:', PROJECT_ROOT)\n"
            "print('Dataset:', DATASET)\n"
        ),
        code(
            "manifest = json.loads((RELEASE_ROOT / 'config' / 'datasets.json').read_text(encoding='utf-8'))\n"
            "info = manifest[DATASET]\n"
            "print(f\"cells={info['cells']:,}, clusters={info['clusters']}\")\n"
            "print(f\"RNA dim={info['rna_dim']}, {info['second_view']} dim={info['second_dim']}\")\n"
            "print('available k:', info['available_k'])\n"
            "print('two-group reference ARI:', info['two_group_best_ari'])\n"
            "print('active clusters:', info['two_group_active_clusters'])\n"
        ),
        markdown("## 1. 输入检查\n\n下面只检查数据、权重和完整命令，不启动训练。"),
        code(
            "dry_command = [\n"
            "    sys.executable, str(RELEASE_ROOT / 'run.py'),\n"
            "    '--dataset', DATASET,\n"
            "    '--profile', PROFILE,\n"
            "    '--dry-run',\n"
            "]\n"
            "subprocess.run(dry_command, cwd=PROJECT_ROOT, check=True)\n"
        ),
        markdown(
            "## 2. 正式训练\n\n"
            "将 `RUN_TRAINING` 改为 `True` 后运行。训练日志、命令与汇总指标会保存在独立输出目录。"
        ),
        code(
            "RUN_TRAINING = False\n"
            "OUTPUT_ROOT = RELEASE_ROOT / 'outputs' / DATASET\n"
            "train_command = [\n"
            "    sys.executable, str(RELEASE_ROOT / 'run.py'),\n"
            "    '--dataset', DATASET,\n"
            "    '--profile', PROFILE,\n"
            "    '--output-root', str(OUTPUT_ROOT),\n"
            "]\n\n"
            "if RUN_TRAINING:\n"
            "    subprocess.run(train_command, cwd=PROJECT_ROOT, check=True)\n"
            "else:\n"
            "    print('训练未启动。确认 dry-run 后，将 RUN_TRAINING 设置为 True。')\n"
            "    print('命令:', subprocess.list2cmdline(train_command))\n"
        ),
        markdown("## 3. 读取结果\n\n训练完成后运行此单元，显示 `summary.json` 中的全部指标。"),
        code(
            "summary_path = OUTPUT_ROOT / 'summary.json'\n"
            "if summary_path.is_file():\n"
            "    rows = json.loads(summary_path.read_text(encoding='utf-8'))\n"
            "    for row in rows:\n"
            "        print(json.dumps(row, ensure_ascii=False, indent=2))\n"
            "else:\n"
            "    print('尚无本次 notebook 输出:', summary_path)\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (sedrenv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, details in DATASETS.items():
        path = TUTORIAL_DIR / details["file"]
        content = notebook(dataset, details["description"])
        path.write_text(
            json.dumps(content, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
