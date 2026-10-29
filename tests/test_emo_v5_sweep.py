"""CPU-only checks for GPU isolation and concurrent sweep scheduling."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_sweep():
    # Load launcher definitions without importing the model stack or .env.
    main_tree = ast.parse((ROOT / 'main.py').read_text())
    names = {'DEFAULT_BASELINE_ARGS', 'EXPERIMENT_SEEDS'}
    definitions = [node for node in main_tree.body
                   if (isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id in names
                       for target in node.targets))
                   or (isinstance(node, ast.FunctionDef) and node.name == 'build_default_argv')]
    namespace = {'__name__': 'sweep_test', '__file__': str(ROOT / 'emo_v5_experiments.py')}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'main.py', 'exec'), namespace)
    tree = ast.parse((ROOT / 'emo_v5_experiments.py').read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module == 'main')]
    exec(compile(tree, 'emo_v5_experiments.py', 'exec'), namespace)
    return namespace


class SweepTests(unittest.TestCase):
    def test_gpu_allocation(self):
        ns = load_sweep()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,5,GPU-abc,7'}):
            self.assertEqual(ns['sweep_gpu_ids'](), ('2', '5', 'GPU-abc'))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,2,5'}):
            self.assertEqual(ns['sweep_gpu_ids'](), ('2', '5'))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaises(RuntimeError):
                ns['sweep_gpu_ids']()

    def test_all_runs_isolated_and_failure_does_not_stall_queue(self):
        ns = load_sweep()
        jobs = list(ns['sweep_jobs']([0]))
        self.assertEqual(len(jobs), 42)
        self.assertEqual(len({record['output_dir'] for _, _, record in jobs}), 42)
        self.assertEqual({record['seed'] for _, _, record in jobs}, {123, 456})
        remaining = {(r['seed'], r['reward_scale'], r['cross_entropy_coefficient'],
                      r['diversity_threshold']) for _, _, r in jobs}
        self.assertFalse(remaining & ns['COMPLETED_COMBINATIONS'])
        self.assertEqual(sum(r[0] == 123 for r in remaining), 15)
        self.assertEqual(sum(r[0] == 456 for r in remaining), 27)
        for threshold in ns['DIVERSITY_THRESHOLDS']:
            self.assertIn((123, 10, 0.05, threshold), remaining)
        lock = threading.Lock()
        barrier = threading.Barrier(3, timeout=10)
        active, seen, calls, snapshots = set(), set(), [], []
        peak = 0

        def fake_run(command, env, check):
            nonlocal peak
            gpu = env['CUDA_VISIBLE_DEVICES']
            i = command.index('--gpu-ids')
            self.assertEqual(command[i + 1], '0')
            self.assertTrue(i + 2 == len(command) or command[i + 2].startswith('--'))
            with lock:
                self.assertNotIn(gpu, active)
                active.add(gpu)
                first = gpu not in seen
                seen.add(gpu)
                calls.append(tuple(command))
                peak = max(peak, len(active))
            if first:
                barrier.wait()
            time.sleep(0.002)
            with lock:
                active.remove(gpu)
            # A failed training run must still release its GPU for the next run.
            return SimpleNamespace(returncode=1)

        coordinator = threading.get_ident()

        def save_json(path, data):
            self.assertEqual(threading.get_ident(), coordinator)
            snapshots.append(len(data['runs']))

        ns['save_json'] = save_json
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,5,7'}), \
             patch.object(ns['subprocess'], 'run', side_effect=fake_run), \
             redirect_stdout(io.StringIO()):
            results = ns['main']()
        self.assertEqual(peak, 3)
        self.assertEqual(seen, {'2', '5', '7'})
        self.assertEqual(len(set(calls)), 42)
        self.assertEqual(len(results), 42)
        self.assertEqual(snapshots, list(range(1, 43)))
        self.assertTrue(all(r['returncode'] == 1 and len(r['gpu_ids']) == 1 for r in results))


if __name__ == '__main__':
    unittest.main()
