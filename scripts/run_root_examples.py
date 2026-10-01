# Run every root-level example script in a fresh interpreter; collect exit codes.
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
EXAMPLES = REPO_ROOT / "examples"
LOG_DIR = HERE.parent / ".workbuddy" / "scratch"
PY = r"D:/anaconda3/envs/py314/python.exe"

full_env = dict(os.environ)
full_env.update({"MPLBACKEND": "Agg", "NUMBA_THREADING_LAYER": "workqueue"})

targets = sorted(p for p in EXAMPLES.glob("*.py") if p.name != "run_all_examples.py")

results = []
for script in targets:
    t0 = time.time()
    proc = subprocess.run(
        [PY, str(script)],
        cwd=str(EXAMPLES),  # scripts may expect their own dir as cwd
        capture_output=True,
        text=True,
        errors="replace",
        env=full_env,
    )
    dt = time.time() - t0
    results.append((script.name, proc.returncode, dt))
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[{status}] {script.name}  exit={proc.returncode}  {dt:.1f}s", flush=True)
    if proc.returncode != 0:
        tail = (proc.stdout or "").strip().splitlines()[-8:]
        err = (proc.stderr or "").strip().splitlines()[-15:]
        print("  --- stdout tail ---")
        print("\n".join("  " + l for l in tail))
        print("  --- stderr tail ---")
        print("\n".join("  " + l for l in err), flush=True)
    else:
        log = LOG_DIR / f"root_example_{script.stem}.log"
        log.write_text((proc.stdout or "") + "\n=====STDERR=====\n" + (proc.stderr or ""),
                       encoding="utf-8")

print("\n===== SUMMARY =====")
fails = [r for r in results if r[1] != 0]
print(f"total={len(results)} pass={len(results)-len(fails)} fail={len(fails)}")
for name, code, dt in fails:
    print(f"  FAIL: {name} exit={code}")
sys.exit(1 if fails else 0)
