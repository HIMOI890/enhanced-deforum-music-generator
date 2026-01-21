"""Automatic1111 extension install hook.

Installs Python deps into the WebUI python environment on startup.
Uses A1111's `launch` helper when available for better compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

def _pip_install(args: list[str]) -> None:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

def main() -> None:
    req = Path(__file__).with_name("requirements.txt")
    if not req.exists():
        return

    lines = [ln.strip() for ln in req.read_text(encoding="utf-8").splitlines()]
    pkgs = [ln for ln in lines if ln and not ln.startswith("#")]

    try:
        import launch  # type: ignore

        for pkg in pkgs:
            name = pkg.split("==")[0].split(">=")[0].strip()
            if launch.is_installed(name):
                continue
            if name == "openai-whisper":
                launch.run_pip("install openai-whisper --no-deps", "deforum-music-extension requirement: openai-whisper")
            else:
                launch.run_pip(f"install {pkg}", f"deforum-music-extension requirement: {pkg}")
        return
    except Exception:
        for pkg in pkgs:
            name = pkg.split("==")[0].split(">=")[0].strip()
            if name == "openai-whisper":
                _pip_install(["openai-whisper", "--no-deps"])
            else:
                _pip_install([pkg])

if __name__ == "__main__":
    main()
