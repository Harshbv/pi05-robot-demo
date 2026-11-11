"""Inline the demo scripts into one Kaggle script: python build.py -> run_all.py"""

import pathlib

here = pathlib.Path(__file__).parent
files = {n: (here / n).read_text(encoding="utf-8") for n in ["droid_demo.py", "libero_demo.py"]}
src = (here / "orchestrate.py").read_text(encoding="utf-8")
src = src.replace("FILES = {}  # filled in by build.py", f"FILES = {files!r}")
(here / "run_all.py").write_text(src, encoding="utf-8")
print("wrote run_all.py")
