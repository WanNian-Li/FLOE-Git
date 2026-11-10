"""Run SAR processing, annotation and optional normalization, without uploads."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

STEP_FILES = {"3": "process_sar.py", "4": "make_labels.py", "5": "normalize.py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default="3,4",
                        help="Ordered subset of 3,4,5; normalization is explicit.")
    parser.add_argument("--class-profile", required=True,
                        choices=("six-class", "hard-ice"),
                        help="Choose retained and target classes explicitly.")
    args = parser.parse_args()
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    if not steps or len(set(steps)) != len(steps) or any(s not in STEP_FILES for s in steps):
        parser.error("--steps must contain unique steps from 3,4,5")
    if steps != sorted(steps):
        parser.error("Steps must follow the order 3,4,5")
    env = os.environ.copy()
    keep, target = (("0,1,2,3,4,5", "1,2,3,4,5")
                    if args.class_profile == "six-class"
                    else ("0,1,2,3", "1,2,3"))
    env["FLOE_KEEP_CLASSES"] = keep
    env["FLOE_TARGET_CLASSES"] = target
    base = Path(__file__).resolve().parent
    for step in steps:
        script = base / STEP_FILES[step]
        print(f"Running step {step}: {script.name}", flush=True)
        result = subprocess.run([sys.executable, str(script)], cwd=base, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
