#!/usr/bin/env python3
"""
scripts/edmg_installer.py

Deterministic installer used by:
- install.ps1 / install.sh
- bootstrap_all.py
- installer_gui.py

This installer *does not* manage GPU drivers. It can, however, install the
appropriate PyTorch wheels (CPU or CUDA) into the EDMG venv.

Examples:
  python scripts/edmg_installer.py install --mode full --backend cpu  --venv venv
  python scripts/edmg_installer.py install --mode full --backend cu121 --venv venv
  python scripts/edmg_installer.py verify
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_windows() -> bool:
    return os.name == "nt"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if _is_windows() else "bin/python")


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> int:
    p = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None)
    return int(p.returncode)


def _pip(py: Path, args: Sequence[str]) -> int:
    return _run([str(py), "-m", "pip", *args], cwd=REPO_ROOT)


def _ensure_venv(venv_dir: Path) -> Path:
    py = _venv_python(venv_dir)
    if py.exists():
        return py
    print(f"[edmg-installer] Creating venv: {venv_dir}")
    if _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=REPO_ROOT) != 0:
        raise RuntimeError("Failed to create venv")
    return _venv_python(venv_dir)


def _select_requirements(mode: str) -> Path:
    candidates = []
    if mode == "minimal":
        candidates.append(REPO_ROOT / "requirements-minimal.txt")
    if mode == "full":
        candidates.append(REPO_ROOT / "requirements-full.txt")
    candidates.append(REPO_ROOT / "requirements.txt")

    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    raise FileNotFoundError("No requirements file found.")


def _torch_index_url(backend: str) -> str:
    backend = backend.strip().lower()
    if backend in {"cpu", "cpu-only"}:
        return "https://download.pytorch.org/whl/cpu"
    if backend in {"cu118", "cu121", "cu124"}:
        return f"https://download.pytorch.org/whl/{backend}"
    raise ValueError(f"Unsupported backend: {backend} (use cpu, cu118, cu121, cu124)")


def _install_torch(py: Path, backend: str) -> int:
    url = _torch_index_url(backend)
    print(f"[edmg-installer] Installing PyTorch ({backend}) from {url}")
    return _pip(
        py,
        [
            "install",
            "-U",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            url,
        ],
    )


def _post_install(py: Path, *, skip_corpora: bool, skip_models: bool) -> None:
    # Best-effort lightweight post install steps.
    if not skip_corpora:
        _run([str(py), "-c", "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"], cwd=REPO_ROOT)
        _run([str(py), "-c", "import spacy; print('spacy ok')"], cwd=REPO_ROOT)

    if not skip_models:
        # Whisper cache corruption happens; keep best-effort.
        _run([str(py), "-c", "import whisper; whisper.load_model('base')"], cwd=REPO_ROOT)


def install(
    *,
    mode: str,
    backend: str,
    venv: Optional[str],
    skip_torch: bool,
    skip_corpora: bool,
    skip_models: bool,
) -> int:
    py = Path(sys.executable)
    if venv:
        py = _ensure_venv(REPO_ROOT / venv)

    if _pip(py, ["install", "-U", "pip", "setuptools", "wheel"]) != 0:
        return 1

    if not skip_torch:
        if _install_torch(py, backend) != 0:
            return 1

    req = _select_requirements(mode)
    print(f"[edmg-installer] Installing requirements from: {req.name}")
    if _pip(py, ["install", "-r", str(req)]) != 0:
        return 1

    # Editable install so `src/` packages are importable everywhere
    if _pip(py, ["install", "-e", "."]) != 0:
        return 1

    _post_install(py, skip_corpora=skip_corpora, skip_models=skip_models)

    print("\n[edmg-installer] OK")
    if venv:
        if _is_windows():
            print(f"  Activate: .\\{venv}\\Scripts\\activate")
        else:
            print(f"  Activate: source ./{venv}/bin/activate")
    print("  Run UI:   python -m enhanced_deforum_music_generator ui --port 7860")
    print("  Verify:   python scripts/edmg_installer.py verify")
    return 0


def verify() -> int:
    code = _run(
        [
            sys.executable,
            "-c",
            "import enhanced_deforum_music_generator as e, deforum_music as d; "
            "print('enhanced_deforum_music_generator:', e.__file__); "
            "print('deforum_music:', d.__file__)",
        ],
        cwd=REPO_ROOT,
    )
    if code != 0:
        return code

    # Verify public API + full Deforum template availability
    code = _run(
        [
            sys.executable,
            "-c",
            "from enhanced_deforum_music_generator.deforum_defaults import make_deforum_settings_template; "
            "d=make_deforum_settings_template(); "
            "print('deforum_template_keys', len(d)); "
            "assert 'W' in d and 'H' in d and 'prompts' in d",
        ],
        cwd=REPO_ROOT,
    )
    return int(code)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="edmg-installer")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("install", help="Install dependencies + editable package")
    pi.add_argument("--mode", default="full", choices=["minimal", "standard", "full", "dev"])
    pi.add_argument("--venv", default="venv", help="Venv dir name (set empty to use current Python)")
    pi.add_argument("--skip-torch", action="store_true", default=False)
    pi.add_argument("--backend", default="cpu", choices=["cpu", "cu118", "cu121", "cu124"])

    # Back-compat flags
    pi.add_argument("--cuda", action="store_true", default=False, help="(deprecated) same as --backend cu121")
    pi.add_argument("--cuda-version", default="", choices=["", "118", "121", "124"], help="(optional) convenience alias")

    pi.add_argument("--skip-corpora", action="store_true", default=False)
    pi.add_argument("--skip-models", action="store_true", default=False)

    pv = sub.add_parser("verify", help="Verify key imports and CLIs")

    args = p.parse_args(argv)

    if args.cmd == "install":
        venv = args.venv.strip() if isinstance(args.venv, str) else "venv"
        if venv == "":
            venv = None

        backend = str(args.backend)
        if args.cuda_version:
            backend = f"cu{args.cuda_version}"
        if bool(args.cuda) and not args.cuda_version and args.backend == "cpu":
            backend = "cu121"

        return install(
            mode=str(args.mode),
            backend=backend,
            venv=venv,
            skip_torch=bool(args.skip_torch),
            skip_corpora=bool(args.skip_corpora),
            skip_models=bool(args.skip_models),
        )

    if args.cmd == "verify":
        return verify()

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
