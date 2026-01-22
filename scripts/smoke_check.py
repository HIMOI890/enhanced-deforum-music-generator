from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> int:
    print("> " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(p.returncode)


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    rc = 0
    rc |= run([sys.executable, "-m", "compileall", "."], root)
    rc |= run([sys.executable, "-m", "pytest", "-q"], root)

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    rc |= run([sys.executable, "-c", "from enhanced_deforum_music_generator.api.main import app; print('routes', len(app.routes))"], root, env=env)
    rc |= run([sys.executable, "-c", "from enhanced_deforum_music_generator.core.prompt_orchestrator import PromptOrchestrator; print('orchestrator ok')"], root, env=env)
    rc |= run([sys.executable, "-c", "from enhanced_deforum_music_generator.core.motion_orchestrator import motion_schedules; print('motion ok')"], root, env=env)

    print("smoke_check rc:", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
