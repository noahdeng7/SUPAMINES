#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
USAGE = """usage: python supamines.py COMMAND [ARGUMENTS]

commands:
  setup                         install dependencies and build the environment
  teacher                       download the BetaTetris teacher
  distill                       train models/distilled_checkpoint.pt
  train NAME                    fine-tune the distilled policy with PPO
  evaluate MODEL [GAMES]        evaluate a checkpoint
  frontier [plan|run|collect]   manage the pace-survival sweep
  play MODEL                    serve a policy to FCEUX
"""


def run(command, cwd=ROOT):
    subprocess.run(command, cwd=cwd, check=True)


def require(args, minimum, maximum):
    if not minimum <= len(args) <= maximum:
        raise SystemExit(USAGE)


def main(args):
    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE, end="")
        return

    command, rest = args[0], args[1:]

    if command == "setup":
        require(rest, 0, 0)
        run([PYTHON, "-m", "pip", "install", "-r", "requirements.txt"])
        run([PYTHON, "setup.py", "build_ext", "--inplace"])
        run([PYTHON, "-m", "pip", "install", "-e", ".", "--no-build-isolation"])
    elif command == "teacher":
        require(rest, 0, 0)
        run([PYTHON, "distillation/download_teacher.py"])
    elif command == "distill":
        require(rest, 0, 0)
        run([PYTHON, "distillation/distill.py"])
    elif command == "train":
        require(rest, 1, 1)
        command = [PYTHON, "train.py", rest[0]]
        checkpoint = ROOT / "models" / "distilled_checkpoint.pt"
        if checkpoint.is_file():
            command += ["--init-model-file", str(checkpoint)]
        run(command, ROOT / "RL")
    elif command == "evaluate":
        require(rest, 1, 2)
        command = [PYTHON, "tools/evaluate.py", rest[0]]
        if len(rest) == 2:
            command += ["-n", str(int(rest[1]))]
        run(command)
    elif command == "frontier":
        require(rest, 0, 1)
        mode = rest[0] if rest else "plan"
        if mode not in {"plan", "run", "collect"}:
            raise SystemExit("frontier mode must be plan, run, or collect")
        run([PYTHON, "tools/frontier.py", mode])
    elif command == "play":
        require(rest, 1, 1)
        run([PYTHON, "demo/fceux.py", rest[0]])
    else:
        raise SystemExit("unknown command: {}\n\n{}".format(command, USAGE))


if __name__ == "__main__":
    main(sys.argv[1:])
