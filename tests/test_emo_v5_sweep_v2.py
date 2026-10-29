"""CPU-only verification of isolated parameter sweeps; no experiments launched."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_sweep():
    main_tree = ast.parse((ROOT / 'main.py').read_text())
    names = {'DEFAULT_BASELINE_ARGS', 'EXPERIMENT_SEEDS'}
    definitions = [node for node in main_tree.body
                   if (isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id in names
                       for target in node.targets))
                   or (isinstance(node, ast.FunctionDef) and node.name == 'build_default_argv')]
    namespace = {'__name__': 'sweep_test', '__file__': str(ROOT / 'emo_v5_experimentsv2.py')}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'main.py', 'exec'), namespace)
    tree = ast.parse((ROOT / 'emo_v5_experimentsv2.py').read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module == 'main')]
    exec(compile(tree, 'emo_v5_experimentsv2.py', 'exec'), namespace)
    return namespace


class SweepV2Tests(unittest.TestCase):
    def test_only_selected_axis_varies_and_cli_matches_main_defaults(self):
        ns = load_sweep()
        jobs = list(ns['sweep_jobs']([0]))
        self.assertEqual(len(jobs), 10 * len(ns['EXPERIMENT_SEEDS']))
        self.assertEqual(len({r['output_dir'] for _, _, r in jobs}), len(jobs))
        self.assertEqual(len({(name, r['seed']) for name, _, r in jobs}), len(jobs))
        defaults = ns['DEFAULT_BASELINE_ARGS']
        for seed in ns['EXPERIMENT_SEEDS']:
            for axis, values in ns['PARAMETER_SWEEPS'].items():
                group = [r for _, _, r in jobs if r['seed'] == seed and r['sweep_parameter'] == axis]
                self.assertEqual([r['sweep_value'] for r in group], list(values))
        for name, argv, record in jobs:
            axis = record['sweep_parameter']
            self.assertIn(axis, name)
            for parameter in ns['FIXED_BASELINE_ARGS']:
                expected = record['sweep_value'] if parameter == axis else defaults[parameter]
                self.assertEqual(record[parameter], expected)
                flag = '--' + parameter.replace('_', '-')
                if expected is None:
                    self.assertNotIn(flag, argv)
                else:
                    self.assertEqual(argv.count(flag), 1)
                    self.assertEqual(argv[argv.index(flag) + 1], str(expected))
            self.assertEqual(argv[argv.index('--seed') + 1], str(record['seed']))
            self.assertEqual(argv[argv.index('--output-dir') + 1], record['output_dir'])
            self.assertEqual(argv[argv.index('--gpu-ids') + 1], '0')
        self.assertNotIn('COMPLETED_COMBINATIONS', ns)
        self.assertEqual(ns['SWEEP_DIR'], Path('outputs/emo_v5/evals/parameter_sweep_v2'))

    def test_gpu_limit_and_missing_allocation(self):
        ns = load_sweep()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,2,5,GPU-abc,7'}):
            self.assertEqual(ns['sweep_gpu_ids'](), ('2', '5'))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '-1'}):
            with self.assertRaisesRegex(RuntimeError, 'sbatch run_experiments.sh'):
                ns['sweep_gpu_ids']()


if __name__ == '__main__':
    unittest.main()
