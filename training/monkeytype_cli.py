import sys
import os
from pathlib import Path


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent / 'MonkeyType'))
    sys.path.append(str(Path(__file__).parent))
    from monkeytype.cli import main
    add_cwd = int(os.environ.get('ADD_CWD_TO_PYTHON_PATH', '1'))
    if add_cwd:
        sys.path.insert(0, os.getcwd())
    sys.exit(main(sys.argv[1:], sys.stdout, sys.stderr))
