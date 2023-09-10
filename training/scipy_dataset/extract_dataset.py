import argparse
from pathlib import Path
import multiprocessing
import subprocess
import os
from itertools import repeat
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests-file', '-i', type=Path, required=True)
    parser.add_argument('--output-dir', '-o', type=Path, required=True)
    parser.add_argument('--timeout', '-t', type=float, default=None)
    parser.add_argument('--num-processes', '-np', type=int, default=None)
    args = parser.parse_args()

    monkeytype_cli_py = (Path(__file__).parent.parent / 'monkeytype_cli.py').resolve()
    assert monkeytype_cli_py.exists()

    test_paths = get_test_paths(args.tests_file)

    with multiprocessing.Pool(args.num_processes) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(
                extract_dataset_from_test,
                zip(repeat(monkeytype_cli_py), test_paths,
                    repeat(args.output_dir), repeat(args.timeout))
            ),
            total=len(test_paths)
        ):
            pass


def get_test_paths(tests_file_path: Path) -> list[str]:
    with open(tests_file_path) as f:
        lines = f.readlines()
    test_paths = set()
    for line in lines:
        line = line.strip()
        if not line.startswith('build/'):
            continue
        tokens = line.split('/')
        assert tokens[5] == 'scipy'
        test_path = '.'.join(tokens[5:-1] + [tokens[-1].split('.')[0]])
        test_paths.add(test_path)
    return list(test_paths)


def extract_dataset_from_test(arguments):
    monkeytype_cli_py, test_path, output_dir, timeout = arguments

    cmd = ['python', str(monkeytype_cli_py), 'run', 'runtests.py', '-t', test_path]

    env = os.environ.copy()
    env['ADD_CWD_TO_PYTHON_PATH'] = '0'
    env['DATASET_OUTPUT_DIR'] = str(output_dir.resolve())

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       check=True, env=env, timeout=timeout, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print(f'Failed to run tests under path: {test_path}\n'
              f'stdout: {e.stdout}\nstderr: {e.stderr}')
    except subprocess.TimeoutExpired:
        print(f'Timeout while running tests under path: {test_path}\n')


if __name__ == '__main__':
    main()
