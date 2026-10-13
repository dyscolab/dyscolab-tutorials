import os
from pathlib import Path
import subprocess
import sys

SEARCH_DIRS = ["examples", "poincare", "simbio", "jablonski"]

# Loads the marimo notebook through its AST representation, evaluates any setup cell
# (including asynchronous setup cells with micropip or environment initialization),
# populates setup globals, and executes the notebook with app.run().
MARIMO_RUNNER = """
import asyncio
import os
import sys
import warnings
from pathlib import Path

# Suppress non-interactive canvas and user warnings from notebook output
warnings.filterwarnings("ignore", message=r".*FigureCanvasAgg is non-interactive.*")
warnings.filterwarnings("ignore", category=UserWarning)

file_path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(file_path.parent))

from marimo._ast.app import InternalApp, _SetupContext
from marimo._ast.load import load_app
from marimo._ast.names import SETUP_CELL_NAME
from marimo._runtime.executor import Evaluator, resolve_executor
from marimo._types.ids import CellId_t

app = load_app(file_path)
if app is None:
    raise RuntimeError(f"Could not load Marimo app from {file_path}")

internal_app = InternalApp(app)
setup_cell = internal_app.graph.cells.get(CellId_t(SETUP_CELL_NAME))
if setup_cell is not None:
    evaluator = Evaluator(executor=resolve_executor(), lifecycles=[])
    glbls = {"__name__": "__main__", "__file__": str(file_path)}
    res = asyncio.run(evaluator.evaluate(setup_cell, glbls))
    if res.exception is not None:
        raise res.exception
    setup_defs = {k: v for k, v in glbls.items() if not k.startswith("__")}
    app._setup = _SetupContext(cell=setup_cell, app=app, hide_code=False)
    app._setup._glbls = setup_defs

app.run()
"""


def find_scripts(directories: list[str]) -> list[Path]:
    scripts = []
    for d in directories:
        dir_path = Path(d)
        if not dir_path.is_dir():
            continue
        scripts.extend(
            p
            for p in dir_path.rglob("*.py")
            if "__pycache__" not in p.parts
        )
    return sorted(set(scripts))


def main():
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"),
        "PYTHONWARNINGS": "ignore::UserWarning",
    }

    if len(sys.argv) > 1:
        targets = [Path(arg) for arg in sys.argv[1:]]
        py_files = []
        for t in targets:
            if t.is_file() and t.suffix == ".py":
                py_files.append(t)
            elif t.is_dir():
                py_files.extend(find_scripts([str(t)]))
        py_files = sorted(set(py_files))
    else:
        py_files = find_scripts(SEARCH_DIRS)

    if not py_files:
        print(f"No Python files found in: {', '.join(SEARCH_DIRS)}")
        return

    failed = []
    for file in py_files:
        print(f"▶ Testing {file}...")
        result = subprocess.run(
            [sys.executable, "-c", MARIMO_RUNNER, str(file)],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"✖ FAILED: {file}\n")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            failed.append(file)
        else:
            print(f"✔ PASSED: {file}\n")

    if failed:
        print(f"Summary: {len(failed)}/{len(py_files)} notebooks failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)

    print(f"Summary: All {len(py_files)} notebooks ran successfully!")


if __name__ == "__main__":
    main()