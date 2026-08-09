"""Run every suite. `python tests/run.py`, or `python tests/run.py chat board` to pick a few.

Each suite is a separate process on purpose: they stub `folder_paths`, `llama_cpp` and parts of the
pack itself, and those stubs must not leak into one another. The exit code is non-zero if anything
failed, so this drops straight into a pre-commit hook or CI.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import COMFY, TESTS  # noqa: E402

# (name, kind, path). JS suites are built and then run — the build step writes run_*.mjs next to it.
SUITES = [
    ("chat", "py", TESTS / "test_chat.py"),
    ("worker", "py", TESTS / "test_worker.py"),
    ("send_image", "py", TESTS / "test_send_image.py"),
    ("dream_board", "py", TESTS / "test_dream_board.py"),
    ("storyboard", "py", TESTS / "test_storyboard.py"),
    ("lora_triggers", "py", TESTS / "test_lora_triggers.py"),
    ("chat.js", "js", TESTS / "js" / "build_chat.mjs"),
    ("dream_board.js", "js", TESTS / "js" / "build_dream_board.mjs"),
]


def _python():
    """ComfyUI's own interpreter. The suites import torch, PIL and comfy, so the system python —
    or the Windows Store shim that answers to `python` on this machine — will not do."""
    for c in (COMFY / ".venv" / "Scripts" / "python.exe", COMFY / ".venv" / "bin" / "python"):
        if c.is_file():
            return str(c)
    return sys.executable


def _run(cmd, cwd):
    # PYTHONIOENCODING: the suites print → ⚠ 🎬, and a Windows console defaults to cp1251.
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("only", nargs="*", help="substrings of suite names to run (default: all)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every check, not just failures")
    args = ap.parse_args()

    suites = [s for s in SUITES if not args.only or any(o in s[0] for o in args.only)]
    if not suites:
        print("no suite matches " + " ".join(args.only))
        return 2
    if any(k == "js" for _, k, _ in suites) and not shutil.which("node"):
        print("! node is not on PATH — the JS suites will be skipped")
        suites = [s for s in suites if s[1] != "js"]

    py = _python()
    print(f"python: {py}\n")
    bad, total = [], 0
    for name, kind, path in suites:
        t0 = time.time()
        if kind == "py":
            rc, out = _run([py, str(path)], COMFY)
        else:
            rc, out = _run(["node", str(path)], path.parent)      # build
            if rc == 0:
                rc, out = _run(["node", str(path.parent / f"run_{path.stem[6:]}.mjs")], path.parent)
        checks = out.count("  ok   ") + out.count("  ok    ")
        total += checks
        ok = rc == 0
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} {checks:>4} checks   {time.time() - t0:5.1f}s")
        if not ok:
            bad.append(name)
        if args.verbose or not ok:
            print("\n".join("        " + ln for ln in out.splitlines() if ln.strip()))
            print()

    print(f"\n{total} checks in {len(suites)} suite(s)")
    if bad:
        print("FAILED: " + ", ".join(bad))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
