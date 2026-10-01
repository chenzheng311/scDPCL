"""Export every training argument, including defaults, without importing torch."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
from pathlib import Path

from config import reproduce

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('two_group',)


def resolve(cli):
    """Run only the shipped trainer's argparse function and its validation."""
    tree = ast.parse((ROOT / 'src/main_dpcl.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
    function.args.args.append(ast.arg(arg='argv'))
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'parse_args':
            node.args = [ast.Name(id='argv', ctx=ast.Load())]
        if isinstance(node, ast.Return):
            node.value = ast.Tuple(elts=[node.value, ast.Name(id='parser', ctx=ast.Load())], ctx=ast.Load())
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {'argparse': argparse}
    exec(compile(module, 'src/main_dpcl.py:parse_args', 'exec'), namespace)
    args, parser = namespace['parse_args'](cli)
    return vars(args), parser


def profile_cli(profile, dataset):
    if profile != 'two_group':
        raise ValueError('Only the current two_group configuration is included.')
    options = list(reproduce.TWO_GROUP_ARGS[dataset])
    return ['--name', dataset, *options, '--output_dir', f'outputs/{profile}/{dataset}']


def snapshot(profile, dataset, cli=None):
    cli = profile_cli(profile, dataset) if cli is None else cli
    values, parser = resolve(cli)
    info = reproduce.load_manifest()[dataset]
    explicit = {token.split('=', 1)[0] for token in cli if token.startswith('--')}
    origins = {}
    defaults = {}
    full_cli = []
    for action in parser._actions:
        if action.dest == 'help':
            continue
        defaults[action.dest] = action.default
        origins[action.dest] = 'explicit' if set(action.option_strings) & explicit else 'trainer_default'
        value = values[action.dest]
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                full_cli.append(option)
        elif value is not None:
            full_cli.extend([option, str(value)])
    ks = [int(k) for k in values['multi_k'].split(',')] if values['multi_k'] else [values['k']]
    derived = {
        'n_clusters': info['clusters'],
        'n_clusters_rule': 'int(max(label) - min(label) + 1)',
        'cells': info['cells'],
        'device': 'cuda if torch.cuda.is_available() else cpu',
        'decode_firnd': not values['raw_decode'],
        'effective_graph_k': ks,
        'effective_equal_graph_weights_when_unspecified': [1 / len(ks)] * len(ks),
        'initialization': 'existing scMDCL checkpoint' if values['scmdcl_init'] else 'rerun scDPCL pretraining, then train',
        'scmdcl_checkpoint': values['scmdcl_pretrain_path'] or f"{values['pretrain_dir']}/{dataset}_pretrain.pkl" if values['scmdcl_init'] else None,
        'optimizer': {'name': 'torch.optim.Adam', 'lr': values['lr'], 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False},
        'training_batch': 'all cells in the dataset (no mini-batch)',
        'dtype': 'float32',
        'cluster_initialization': {'name': 'sklearn.cluster.KMeans', 'n_init': 10, 'init': 'k-means++', 'max_iter': 300, 'tol': 1e-4, 'algorithm': 'lloyd', 'random_state': None if values['no_seed'] else values['seed']},
        'student_t_alpha': 1,
        'firnd_steps': 1,
        'best_epoch_selection': 'maximum training evaluation ARI using reference labels',
        'final_trained_state_dict_available_in_archive': False,
    }
    return {'profile': profile, 'dataset': dataset, 'cli_parameters': values, 'parameter_origin': origins, 'trainer_defaults': defaults, 'derived_and_implementation_settings': derived, 'configured_argv': cli, 'full_argv': full_cli, 'command': subprocess.list2cmdline(['python', 'src/main_dpcl.py', *full_cli])}


def write_exports():
    all_rows = []
    documents = {}
    for profile in PROFILES:
        destination = ROOT / 'config/parameters' / profile
        destination.mkdir(parents=True, exist_ok=True)
        commands = ['# Run from the release root. All non-boolean/default values are explicit.']
        for dataset in reproduce.ALL_DATASETS:
            document = snapshot(profile, dataset)
            documents[(profile, dataset)] = document
            (destination / f'{dataset}.json').write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            commands.append(document['command'])
            for key, value in document['cli_parameters'].items():
                all_rows.append({'profile': profile, 'dataset': dataset, 'parameter': key, 'value': json.dumps(value, ensure_ascii=False), 'origin': document['parameter_origin'][key]})
        (destination / 'commands.txt').write_text('\n\n'.join(commands) + '\n', encoding='utf-8')
    with (ROOT / 'config/ALL_PARAMETERS.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    datasets = reproduce.ALL_DATASETS
    lines = ['# scDPCL 完整参数', '', '默认方案：`two_group`，与 2026-10-01 的 PBMC-10k 实际运行命令一致。', '',
             '本表从随包 `src/main_dpcl.py` 的参数解析函数与 `config/reproduce.py` 自动生成，包含所有显式值及隐式默认值。路径以本文件夹为运行根目录；表中的输出路径用于示例，正式运行采用时间戳目录。', '',
             '`config/parameters/<profile>/<dataset>.json` 包含逐项来源、原始默认值、完整 argv、派生参数与实现固定设置；`config/ALL_PARAMETERS.csv` 仅汇总当前方案的三个数据集配置。', '',
             '| 参数 | PBMC-10k | PBMC-3k | BMNC |', '|---|---|---|---|']
    for key in documents[('two_group', datasets[0])]['cli_parameters']:
        row = [json.dumps(documents[('two_group', dataset)]['cli_parameters'][key], ensure_ascii=False) for dataset in datasets]
        lines.append('| `' + key + '` | ' + ' | '.join('`' + value + '`' for value in row) + ' |')
    lines.extend(['', '## 实现中固定或数据派生的设置', '',
                  '- 类别数按标签的最大值减最小值加 1 得到：PBMC-10k=19、PBMC-3k=16、BMNC=27。',
                  '- 全图训练，float32；编码维度为 输入→256→128→20，解码维度为 20→128→256→输入。',
                  '- Adam：betas=(0.9, 0.999)，eps=1e-8，weight_decay=0，amsgrad=False。默认不启用学习率调度。',
                  '- KMeans：n_init=10，random_state=0；其余使用归档环境 scikit-learn 1.7.0 的默认值（k-means++、max_iter=300、tol=1e-4、lloyd）。',
                  '- Student-t 分配 alpha=1；FIRND 传播为一次邻接矩阵乘法；多图未指定权重时等权平均。',
                  '- 预训练日程：每个视图各 50 轮重构，100 轮融合；正式训练 500 轮（日志 epoch 从 0 计数）。PBMC-10k 使用已有 scMDCL 权重，不重新预训练。',
                  '- PBMC-10k 默认 n_private=0；全部默认 two_group 配置 beta=gamma=0，相关损失虽然保留在模型中，但权重为零。',
                  '- 最佳 epoch 依据真实标签 ARI 选择；恢复步骤受空缺簇数限制，不能把 best/恢复结果称为无标签模型选择。',
                  '- 全部模型数值常量及损失实现以随包源码为准；本表覆盖全部 CLI 参数，并补充主要固定设置。', '',
                  '重新导出：`python -m config.export_parameters`。'])
    (ROOT / 'FULL_PARAMETERS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Exported {len(documents)} configurations, {len(all_rows) // len(documents)} CLI parameters each.')


if __name__ == '__main__':
    write_exports()
