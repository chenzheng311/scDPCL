"""Validate the standalone archive without running full-data training."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-hashes', action='store_true')
    args = parser.parse_args()
    (ROOT / 'verification').mkdir(exist_ok=True)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(ROOT), 'full_training_rerun': False, 'checks': []}
    if args.check_hashes:
        manifest = json.loads((ROOT / 'FILES_SHA256.json').read_text(encoding='utf-8'))
        for item in manifest['files']:
            path = ROOT / item['path']
            assert path.is_file(), f'Missing file: {path}'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], f'Hash mismatch: {path}'
        report['checks'].append({'check': 'delivery_hashes', 'files': len(manifest['files'])})

    sources = list(ROOT.rglob('*.py'))
    for source in sources:
        ast.parse(source.read_text(encoding='utf-8-sig'), filename=str(source))
    report['checks'].append({'check': 'python_syntax', 'files': len(sources)})

    import numpy as np
    import scipy.sparse as sp
    import torch
    from config.export_parameters import PROFILES, profile_cli, resolve, snapshot
    from config.reproduce import ALL_DATASETS, load_manifest, required_files
    sys.path.insert(0, str(ROOT / 'src'))
    import main_dpcl as trainer
    assert Path(trainer.__file__).resolve().is_relative_to(ROOT)
    manifest = load_manifest()
    for dataset, info in manifest.items():
        data = ROOT / 'input' / dataset
        labels = np.load(data / 'label.npy', allow_pickle=False)
        assert len(labels) == info['cells']
        assert int(labels.max() - labels.min() + 1) == info['clusters']
        assert sorted(np.unique(labels, return_counts=True)[1].tolist()) == sorted(info['class_counts'])
        for view, dimension in (('RNA', info['rna_dim']), ('ATAC', info['second_dim'])):
            matrix = np.load(data / f'{view}_fea.npy', allow_pickle=False)
            assert matrix.shape == (info['cells'], dimension)
            assert np.isfinite(matrix).all()
            for k in info['available_k']:
                graph = sp.load_npz(data / f'{view}_euc_{k}.npz')
                assert graph.shape == (info['cells'], info['cells'])
                assert np.isfinite(graph.data).all()
        report['checks'].append({'check': 'input_data', 'dataset': dataset, 'cells': len(labels), 'clusters': len(np.unique(labels))})

    for profile in PROFILES:
        for dataset in ALL_DATASETS:
            for path in required_files(dataset, manifest[dataset], 'optimized', profile):
                assert path.is_file(), path
            exported = json.loads((ROOT / 'config/parameters' / profile / f'{dataset}.json').read_text(encoding='utf-8'))
            assert exported == snapshot(profile, dataset), f'Stale parameter export: {profile}/{dataset}'
            cli = profile_cli(profile, dataset)
            expected, _ = resolve(cli)
            old_argv = sys.argv
            try:
                sys.argv = [str(ROOT / 'src/main_dpcl.py'), *cli]
                actual = trainer.parse_args()
            finally:
                sys.argv = old_argv
            assert vars(actual) == expected
            old_argv = sys.argv
            try:
                sys.argv = ['main_dpcl.py', *exported['full_argv']]
                assert vars(trainer.parse_args()) == expected
            finally:
                sys.argv = old_argv
            actual.device = torch.device('cpu')
            actual.n_clusters = manifest[dataset]['clusters']
            trainer.setup_seed(actual.seed)
            model = trainer.build_model(actual)
            if actual.scmdcl_init:
                state = torch.load(ROOT / 'model_pretrained' / f'{dataset}_pretrain.pkl', map_location='cpu', weights_only=True)
                compatible = {k: v for k, v in state.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
                assert compatible and any(k.startswith('gae1.') for k in compatible) and any(k.startswith('gae2.') for k in compatible)
                model.load_state_dict(compatible, strict=False)
            else:
                filename = Path(trainer.pretrain_path(actual)).name
                state = torch.load(ROOT / 'model_pretrained' / filename, map_location='cpu', weights_only=True)
                model.load_state_dict(state, strict=True)
                compatible = state
            model.eval()
            x1, x2 = torch.randn(8, actual.n_d1), torch.randn(8, actual.n_d2)
            adj = torch.eye(8).to_sparse()
            outputs = model(x1, adj, x2, adj)
            assert len(outputs) == 12
            assert outputs[0].shape == x1.shape and outputs[2].shape == x2.shape
            for pair in outputs[4:6]:
                for q in pair:
                    assert q.shape == (8, actual.n_clusters) and torch.isfinite(q).all()
                    assert torch.allclose(q.sum(1), torch.ones(8), atol=1e-5)
            loss = (outputs[0] - x1).square().mean() + (outputs[2] - x2).square().mean()
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            report['checks'].append({'check': 'parameters_checkpoint_forward_backward', 'profile': profile, 'dataset': dataset, 'cli_parameters': len(expected), 'loaded_tensors': len(compatible), 'synthetic_cells': 8})

    for mode, profile in [('optimized', p) for p in PROFILES] + [('original', 'two_group')]:
        command = [sys.executable, str(ROOT / 'run.py'), '--dataset', 'all', '--profile', profile, '--mode', mode, '--dry-run', '--reuse-original-pretrain']
        result = subprocess.run(command, cwd=tempfile.gettempdir(), text=True, encoding='utf-8', errors='replace', capture_output=True)
        assert result.returncode == 0 and 'Dry run passed' in result.stdout, result.stdout + result.stderr
        (ROOT / 'verification' / f'dry_run_{mode}_{profile}.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        report['checks'].append({'check': 'dry_run_from_external_cwd', 'mode': mode, 'profile': profile, 'datasets': list(ALL_DATASETS)})

    # Verify the copied historical predictions against the labels and logged ARIs.
    from sklearn.metrics import adjusted_rand_score
    references = [
        ('PBMC-10k', 'seed0_label.npy', 0.914055),
        ('PBMC-10k', 'seed0_label_recovered.npy', 0.920613),
        ('PBMC-3k', 'seed0_label.npy', 0.625453),
        ('PBMC-3k', 'seed0_label_recovered.npy', 0.677099),
        ('BMNC', 'seed0_label.npy', 0.742072),
    ]
    for dataset, filename, expected_ari in references:
        prediction_path = ROOT / 'reference_results' / dataset / filename
        y = np.load(ROOT / 'input' / dataset / 'label.npy')
        prediction = np.load(prediction_path)
        ari = adjusted_rand_score(y, prediction)
        assert abs(ari - expected_ari) <= 5.1e-7, (prediction_path, ari, expected_ari)
        report['checks'].append({'check': 'historical_prediction_ari', 'path': prediction_path.relative_to(ROOT).as_posix(), 'ari': ari, 'active_clusters': len(np.unique(prediction))})

    report['status'] = 'passed'
    report_path = ROOT / 'verification/report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Passed {len(report['checks'])} checks. Full training was not rerun. Report: {report_path}")


if __name__ == '__main__':
    main()
