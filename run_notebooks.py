"""Execute notebooks in place (cell outputs saved). Usage:
    python run_notebooks.py 01 02 ...   # run specific notebooks
    python run_notebooks.py all         # run every notebook 00..08
"""
import os
import sys
import glob
import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(ROOT, "notebooks")


def run_one(path):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                            resources={"metadata": {"path": ROOT}})
    client.execute()
    nbformat.write(nb, path)
    print("OK  ", os.path.basename(path))


def main(args):
    files = sorted(glob.glob(os.path.join(NB_DIR, "*.ipynb")))
    if args and args[0] != "all":
        files = [f for f in files if any(os.path.basename(f).startswith(a) for a in args)]
    for f in files:
        try:
            run_one(f)
        except Exception as e:
            print("FAIL", os.path.basename(f), "->", type(e).__name__, str(e)[:500])
            raise


if __name__ == "__main__":
    main(sys.argv[1:])
