import argparse
import json
import os
import pathlib
import shutil
import subprocess
from typing import List, Union


def copy_results(
    detectors_names: List[str], nab_dir: pathlib.Path, results_dir: pathlib.Path,
) -> None:
    target_dir = nab_dir / "results"
    for result in results_dir.iterdir():
        if not result.is_dir():
            continue
        if (target_dir / result.name).exists():
            continue
        if result.name not in detectors_names:
            continue
        print(f"Moving files {result} -> {target_dir / result.name} ...")
        shutil.copytree(result, target_dir / result.name)


def update_nab_config(
        detectors_names: List[str], nab_dir: pathlib.Path) -> None:
    nab_config = nab_dir / "config" / "thresholds.json"
    threshold_record = {
        "reward_low_FN_rate": {
            "threshold": 0.5
        },
        "reward_low_FP_rate": {
            "threshold": 0.5
        },
        "standard": {
            "threshold": 0.5
        }
    }
    config = json.loads(nab_config.read_text())
    for detector in detectors_names:
        if detector not in config:
            print(f"Adding thresholds for {detector} ...")
            config[detector] = threshold_record
    nab_config.write_text(json.dumps(config))


def run_nab(
    python: str, detectors_names: List[str], nab_dir: Union[pathlib.Path, str],
) -> None:
    cmd = [
        python,
        'run.py',
        '-d',
        ','.join(detectors_names),
        '--score',
        '--normalize'
    ]
    print(' '.join(cmd))
    os.chdir(nab_dir)
    subprocess.call(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run NAB Benchmark on results files.',
    )

    parser.add_argument('nab_path', type=str, help='path with nab package')
    parser.add_argument('results', type=str, help='path with results csv files')
    parser.add_argument('-d', '--detectors', nargs='+', default=[])
    parser.add_argument('--python_path', type=str, default='python',
                        required=False, help='path to python executable')

    args = parser.parse_args()

    detectors = args.detectors
    python_path = args.python_path
    nab_root = pathlib.Path(args.nab_path).absolute()
    results_root = pathlib.Path(args.results)

    print(f"Starting NAB Evaluation using {nab_root}.")
    print(f"Reading results from {results_root} ...")
    copy_results(detectors, nab_root, results_root)

    print("Checking NAB config file ...")
    update_nab_config(detectors, nab_root)
    print("Config check done.")

    print("Running evaluation:")
    run_nab(python_path, detectors, nab_root)
