"""The docs must still describe the nodes the pack registers.

A thin wrapper around ``tools/gen_readme_index.py --check`` so the audit runs with everything
else: the README's generated index is regenerated in memory and compared, every node folder must
have a section in ``docs/``, every node must be written about, every local link and anchor must
resolve, and every backticked `snake_case` term must be a real input/output (or listed in
``tools/known_terms.txt``). It prints the same ``  ok   `` lines the other suites do, so run.py
counts its checks too.

Nothing is written: `--check` only reads. Run the tool without the flag to fix what it reports.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import PACK  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "gen_readme_index", str(PACK / "tools" / "gen_readme_index.py"))
gen = importlib.util.module_from_spec(spec)
sys.modules["gen_readme_index"] = gen
spec.loader.exec_module(gen)

sys.argv = ["gen_readme_index.py", "--check"]
sys.exit(gen.main())
