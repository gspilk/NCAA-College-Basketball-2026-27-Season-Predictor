"""
run_all.py
Runs every step in order: regular season, postseason, conference leaders, rankings.
"""
import os
import subprocess
import sys

root = os.path.dirname(os.path.abspath(__file__))
steps = ["season.py", "postseason.py", "conf_leaders.py", "rankings.py"]

for step in steps:
    print(f"\n========== {step} ==========")
    result = subprocess.run([sys.executable, os.path.join(root, step)], cwd=root)
    if result.returncode != 0:
        print(f"\n{step} failed -- stopping here.")
        break