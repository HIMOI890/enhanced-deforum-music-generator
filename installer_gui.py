#!/usr/bin/env python3
"""
installer_gui.py

EDMG "install everything" GUI installer & launcher.

What it can do:
- Standalone EDMG (repo venv + dependencies) with CPU or CUDA PyTorch wheels
- Optional external installs:
  - Automatic1111 stable-diffusion-webui
    - Deforum extension
    - EDMG A1111 extension bundle (this repo's /a1111_extension)
  - ComfyUI (with its own .venv)

Notes:
- This does NOT install GPU drivers. It only installs Python wheels.
- For A1111, first launch can take a while (it creates its own venv and downloads deps).
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, Optional, Sequence


# -------------------------
# Constants / Defaults
# -------------------------

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
EXTERNAL_DIR_DEFAULT = REPO_ROOT / "external"

A1111_REPO = "https://github.com/AUTOMATIC1111/stable-diffusion-webui.git"
DEFORUM_REPO = "https://github.com/deforum/sd-webui-deforum.git"
COMFYUI_REPO = "https://github.com/Comfy-Org/ComfyUI.git"

DEFAULT_EDMG_PORT = 7860
DEFAULT_A1111_PORT = 7861  # avoid conflict with EDMG UI
DEFAULT_COMFYUI_PORT = 8188

BACKEND_CHOICES = ["cpu", "cu118", "cu121", "cu124"]
MODE_CHOICES = ["minimal", "standard", "full", "dev"]


def is_windows() -> bool:
    return os.name == "nt"


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if is_windows() else "bin/python")


def edmg_venv_dir() -> Path:
    return REPO_ROOT / "venv"


# -------------------------
# Logging + subprocess helpers
# -------------------------

def log_put(q: "queue.Queue[str]", msg: str) -> None:
    q.put(msg.rstrip("\n"))


def run_stream(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    logq: Optional["queue.Queue[str]"] = None,
) -> int:
    """Run a command and stream stdout/stderr to logq."""
    if logq:
        log_put(logq, f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if logq:
            log_put(logq, line.rstrip("\n"))
    proc.wait()
    return int(proc.returncode)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def git_clone_or_pull(repo_url: str, dest: Path, logq: "queue.Queue[str]") -> None:
    ensure_dir(dest.parent)
    if (dest / ".git").exists():
        log_put(logq, f"[git] Updating: {dest}")
        run_stream(["git", "-C", str(dest), "pull", "--rebase"], logq=logq)
    else:
        log_put(logq, f"[git] Cloning: {repo_url} -> {dest}")
        run_stream(["git", "clone", repo_url, str(dest)], logq=logq)


def backend_to_torch_index(backend: str) -> str:
    backend = backend.strip().lower()
    if backend == "cpu":
        return "https://download.pytorch.org/whl/cpu"
    if backend in {"cu118", "cu121", "cu124"}:
        return f"https://download.pytorch.org/whl/{backend}"
    raise ValueError(f"Unsupported backend: {backend}")


# -------------------------
# Install routines
# -------------------------

def install_edmg(mode: str, backend: str, logq: "queue.Queue[str]") -> None:
    log_put(logq, f"[EDMG] Installing standalone environment (mode={mode}, backend={backend})")
    installer = SCRIPTS_DIR / "edmg_installer.py"
    if not installer.exists():
        raise FileNotFoundError(f"Missing installer: {installer}")

    rc = run_stream(
        [
            sys.executable,
            str(installer),
            "install",
            "--mode",
            mode,
            "--backend",
            backend,
            "--venv",
            "venv",
        ],
        cwd=REPO_ROOT,
        logq=logq,
    )
    if rc != 0:
        raise RuntimeError(f"EDMG install failed (exit={rc})")


def install_a1111(
    *,
    a1111_dir: Path,
    backend: str,
    port: int,
    install_deforum: bool,
    install_edmg_extension: bool,
    logq: "queue.Queue[str]",
) -> None:
    if not git_available():
        raise RuntimeError("Git is required to install A1111/Deforum. Please install Git and re-run.")

    log_put(logq, f"[A1111] Installing/Updating in: {a1111_dir}")
    git_clone_or_pull(A1111_REPO, a1111_dir, logq)

    ext_dir = a1111_dir / "extensions"
    ensure_dir(ext_dir)

    if install_deforum:
        deforum_dest = ext_dir / "sd-webui-deforum"
        log_put(logq, f"[A1111] Installing Deforum extension -> {deforum_dest}")
        git_clone_or_pull(DEFORUM_REPO, deforum_dest, logq)

    if install_edmg_extension:
        src_ext = REPO_ROOT / "a1111_extension"
        if not src_ext.exists():
            raise FileNotFoundError("Missing /a1111_extension in this repo.")
        dest_ext = ext_dir / "enhanced-deforum-music-generator"
        log_put(logq, f"[A1111] Installing EDMG extension bundle -> {dest_ext}")
        if dest_ext.exists():
            shutil.rmtree(dest_ext)
        shutil.copytree(src_ext, dest_ext)

    # Create a dedicated launcher script so we don't overwrite user's webui-user.*
    torch_index = backend_to_torch_index(backend)
    if is_windows():
        launcher = a1111_dir / "webui-user-edmg.bat"
        cmd_args = f'--api --listen --port {port}'
        if backend == "cpu":
            cmd_args = f'--use-cpu all --precision full --no-half --skip-torch-cuda-test {cmd_args}'
        content = (
            "@echo off\n"
            "set PYTHON=\n"
            "set GIT=\n"
            "set VENV_DIR=venv\n"
            f"set COMMANDLINE_ARGS={cmd_args}\n"
            f"set TORCH_COMMAND=pip install torch torchvision torchaudio --index-url {torch_index}\n"
            "call webui.bat\n"
        )
        launcher.write_text(content, encoding="utf-8")
    else:
        launcher = a1111_dir / "webui-user-edmg.sh"
        cmd_args = f'--api --listen --port {port}'
        if backend == "cpu":
            cmd_args = f'--use-cpu all --precision full --no-half --skip-torch-cuda-test {cmd_args}'
        content = (
            "#!/usr/bin/env bash\n"
            "set -e\n"
            f'export COMMANDLINE_ARGS="{cmd_args}"\n'
            f'export TORCH_COMMAND="pip install torch torchvision torchaudio --index-url {torch_index}"\n'
            "bash webui.sh\n"
        )
        launcher.write_text(content, encoding="utf-8")
        launcher.chmod(0o755)

    log_put(logq, f"[A1111] Launcher created: {launcher.name}")
    log_put(logq, "[A1111] First launch may take a while (creates venv + downloads).")


def install_comfyui(
    *,
    comfy_dir: Path,
    backend: str,
    logq: "queue.Queue[str]",
) -> None:
    if not git_available():
        raise RuntimeError("Git is required to install ComfyUI. Please install Git and re-run.")

    log_put(logq, f"[ComfyUI] Installing/Updating in: {comfy_dir}")
    git_clone_or_pull(COMFYUI_REPO, comfy_dir, logq)

    # Create venv inside ComfyUI folder
    venv_dir = comfy_dir / ".venv"
    py = venv_python(venv_dir)
    if not py.exists():
        log_put(logq, f"[ComfyUI] Creating venv: {venv_dir}")
        run_stream([sys.executable, "-m", "venv", str(venv_dir)], cwd=comfy_dir, logq=logq)

    # Upgrade pip
    run_stream([str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], cwd=comfy_dir, logq=logq)

    # Install torch
    torch_index = backend_to_torch_index(backend)
    log_put(logq, f"[ComfyUI] Installing torch ({backend})")
    run_stream(
        [str(py), "-m", "pip", "install", "-U", "torch", "torchvision", "torchaudio", "--index-url", torch_index],
        cwd=comfy_dir,
        logq=logq,
    )

    # Install requirements
    req = comfy_dir / "requirements.txt"
    if req.exists():
        log_put(logq, "[ComfyUI] Installing requirements.txt")
        run_stream([str(py), "-m", "pip", "install", "-r", str(req)], cwd=comfy_dir, logq=logq)

    log_put(logq, "[ComfyUI] OK")


# -------------------------
# Launcher routines
# -------------------------

def start_process(cmd: Sequence[str], *, cwd: Optional[Path], logq: "queue.Queue[str]") -> subprocess.Popen:
    log_put(logq, f"[start] {' '.join(cmd)}")
    return subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )


# -------------------------
# Tk UI
# -------------------------

def main() -> None:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("EDMG Installer & Launcher")

    logq: "queue.Queue[str]" = queue.Queue()
    procs: Dict[str, subprocess.Popen] = {}

    # State vars
    backend_var = tk.StringVar(value="cpu")
    mode_var = tk.StringVar(value="full")

    external_dir_var = tk.StringVar(value=str(EXTERNAL_DIR_DEFAULT))
    a1111_dir_var = tk.StringVar(value=str(EXTERNAL_DIR_DEFAULT / "stable-diffusion-webui"))
    comfy_dir_var = tk.StringVar(value=str(EXTERNAL_DIR_DEFAULT / "ComfyUI"))

    install_standalone_var = tk.BooleanVar(value=True)
    install_a1111_var = tk.BooleanVar(value=False)
    install_deforum_var = tk.BooleanVar(value=True)
    install_edmg_ext_var = tk.BooleanVar(value=True)
    install_comfyui_var = tk.BooleanVar(value=False)

    edmg_port_var = tk.IntVar(value=DEFAULT_EDMG_PORT)
    a1111_port_var = tk.IntVar(value=DEFAULT_A1111_PORT)
    comfy_port_var = tk.IntVar(value=DEFAULT_COMFYUI_PORT)

    # Widgets
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    tab_install = ttk.Frame(nb)
    tab_launch = ttk.Frame(nb)
    tab_logs = ttk.Frame(nb)
    nb.add(tab_install, text="Install")
    nb.add(tab_launch, text="Launch")
    nb.add(tab_logs, text="Logs")

    # ----- Logs tab -----
    txt = tk.Text(tab_logs, height=30, wrap="word")
    txt.pack(fill="both", expand=True, side="left")
    sb = ttk.Scrollbar(tab_logs, command=txt.yview)
    sb.pack(side="right", fill="y")
    txt.configure(yscrollcommand=sb.set)

    def pump_logs() -> None:
        try:
            while True:
                line = logq.get_nowait()
                txt.insert("end", line + "\n")
                txt.see("end")
        except queue.Empty:
            pass
        root.after(150, pump_logs)

    root.after(150, pump_logs)

    # ----- Install tab -----
    def browse_dir(var: tk.StringVar) -> None:
        p = filedialog.askdirectory(initialdir=var.get() or str(REPO_ROOT))
        if p:
            var.set(p)

    row = 0
    ttk.Label(tab_install, text="Compute Backend:").grid(row=row, column=0, sticky="w", padx=8, pady=6)
    backend_box = ttk.Combobox(tab_install, textvariable=backend_var, values=BACKEND_CHOICES, state="readonly", width=10)
    backend_box.grid(row=row, column=1, sticky="w", padx=8, pady=6)
    ttk.Label(tab_install, text="EDMG Mode:").grid(row=row, column=2, sticky="w", padx=8, pady=6)
    mode_box = ttk.Combobox(tab_install, textvariable=mode_var, values=MODE_CHOICES, state="readonly", width=10)
    mode_box.grid(row=row, column=3, sticky="w", padx=8, pady=6)

    row += 1
    targets = ttk.LabelFrame(tab_install, text="Install Targets")
    targets.grid(row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=6)
    targets.columnconfigure(1, weight=1)

    ttk.Checkbutton(targets, text="Standalone EDMG (venv + deps)", variable=install_standalone_var).grid(
        row=0, column=0, sticky="w", padx=8, pady=4
    )
    ttk.Checkbutton(targets, text="Automatic1111 (stable-diffusion-webui)", variable=install_a1111_var).grid(
        row=1, column=0, sticky="w", padx=8, pady=4
    )
    ttk.Checkbutton(targets, text="  ├─ Deforum extension", variable=install_deforum_var).grid(
        row=2, column=0, sticky="w", padx=22, pady=2
    )
    ttk.Checkbutton(targets, text="  └─ EDMG A1111 extension bundle", variable=install_edmg_ext_var).grid(
        row=3, column=0, sticky="w", padx=22, pady=2
    )
    ttk.Checkbutton(targets, text="ComfyUI", variable=install_comfyui_var).grid(
        row=4, column=0, sticky="w", padx=8, pady=4
    )

    row += 1
    paths = ttk.LabelFrame(tab_install, text="Paths")
    paths.grid(row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=6)
    paths.columnconfigure(1, weight=1)

    ttk.Label(paths, text="External folder:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
    ttk.Entry(paths, textvariable=external_dir_var).grid(row=0, column=1, sticky="ew", padx=8, pady=4)
    ttk.Button(paths, text="Browse", command=lambda: browse_dir(external_dir_var)).grid(row=0, column=2, padx=8, pady=4)

    ttk.Label(paths, text="A1111 folder:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
    ttk.Entry(paths, textvariable=a1111_dir_var).grid(row=1, column=1, sticky="ew", padx=8, pady=4)
    ttk.Button(paths, text="Browse", command=lambda: browse_dir(a1111_dir_var)).grid(row=1, column=2, padx=8, pady=4)

    ttk.Label(paths, text="ComfyUI folder:").grid(row=2, column=0, sticky="w", padx=8, pady=4)
    ttk.Entry(paths, textvariable=comfy_dir_var).grid(row=2, column=1, sticky="ew", padx=8, pady=4)
    ttk.Button(paths, text="Browse", command=lambda: browse_dir(comfy_dir_var)).grid(row=2, column=2, padx=8, pady=4)

    row += 1
    ports = ttk.LabelFrame(tab_install, text="Ports")
    ports.grid(row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=6)
    ttk.Label(ports, text="EDMG UI:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
    ttk.Entry(ports, textvariable=edmg_port_var, width=8).grid(row=0, column=1, sticky="w", padx=8, pady=4)
    ttk.Label(ports, text="A1111:").grid(row=0, column=2, sticky="w", padx=8, pady=4)
    ttk.Entry(ports, textvariable=a1111_port_var, width=8).grid(row=0, column=3, sticky="w", padx=8, pady=4)
    ttk.Label(ports, text="ComfyUI:").grid(row=0, column=4, sticky="w", padx=8, pady=4)
    ttk.Entry(ports, textvariable=comfy_port_var, width=8).grid(row=0, column=5, sticky="w", padx=8, pady=4)

    row += 1
    btns = ttk.Frame(tab_install)
    btns.grid(row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=10)

    install_btn = ttk.Button(btns, text="Install Selected", width=22)
    install_btn.pack(side="left", padx=6)
    verify_btn = ttk.Button(btns, text="Verify EDMG venv", width=18)
    verify_btn.pack(side="left", padx=6)

    def do_install() -> None:
        backend = backend_var.get().strip()
        mode = mode_var.get().strip()
        # Update derived default paths if user changed external dir
        ext_root = Path(external_dir_var.get())
        a1111_path = Path(a1111_dir_var.get())
        comfy_path = Path(comfy_dir_var.get())

        ensure_dir(ext_root)

        try:
            if install_standalone_var.get():
                install_edmg(mode, backend, logq)

            if install_a1111_var.get():
                install_a1111(
                    a1111_dir=a1111_path,
                    backend=backend,
                    port=int(a1111_port_var.get()),
                    install_deforum=bool(install_deforum_var.get()),
                    install_edmg_extension=bool(install_edmg_ext_var.get()),
                    logq=logq,
                )

            if install_comfyui_var.get():
                install_comfyui(comfy_dir=comfy_path, backend=backend, logq=logq)

            log_put(logq, "[DONE] All selected installs completed.")
        except Exception as e:
            log_put(logq, f"[ERROR] {e}")
            log_put(logq, "See logs for details.")

    def run_in_worker(fn):
        install_btn.config(state="disabled")
        verify_btn.config(state="disabled")

        def _worker():
            try:
                fn()
            finally:
                install_btn.config(state="normal")
                verify_btn.config(state="normal")

        threading.Thread(target=_worker, daemon=True).start()

    install_btn.configure(command=lambda: run_in_worker(do_install))

    def do_verify() -> None:
        py = venv_python(edmg_venv_dir())
        if not py.exists():
            log_put(logq, "[verify] EDMG venv not found; run install first.")
            return
        rc = run_stream([str(py), str(SCRIPTS_DIR / "edmg_installer.py"), "verify"], cwd=REPO_ROOT, logq=logq)
        log_put(logq, f"[verify] exit={rc}")

    verify_btn.configure(command=lambda: run_in_worker(do_verify))

    # ----- Launch tab -----
    launch = ttk.Frame(tab_launch)
    launch.pack(fill="both", expand=True, padx=10, pady=10)

    ttk.Label(launch, text="Launchers").grid(row=0, column=0, columnspan=3, sticky="w", pady=4)

    def start_edmg_ui():
        py = venv_python(edmg_venv_dir())
        if not py.exists():
            messagebox.showerror("EDMG", "EDMG venv not found. Run install first.")
            return
        port = int(edmg_port_var.get())
        procs["edmg_ui"] = start_process([str(py), "-m", "enhanced_deforum_music_generator", "ui", "--port", str(port)], cwd=REPO_ROOT, logq=logq)
        webbrowser.open(f"http://127.0.0.1:{port}")

    def start_a1111_ui():
        a1111_path = Path(a1111_dir_var.get())
        if not a1111_path.exists():
            messagebox.showerror("A1111", "A1111 folder not found. Install first.")
            return
        if is_windows():
            launcher = a1111_path / "webui-user-edmg.bat"
            cmd = ["cmd.exe", "/c", str(launcher)]
        else:
            launcher = a1111_path / "webui-user-edmg.sh"
            cmd = ["bash", str(launcher)]
        procs["a1111"] = start_process(cmd, cwd=a1111_path, logq=logq)
        webbrowser.open(f"http://127.0.0.1:{int(a1111_port_var.get())}")

    def start_comfyui_ui():
        comfy_path = Path(comfy_dir_var.get())
        venv_dir = comfy_path / ".venv"
        py = venv_python(venv_dir)
        if not py.exists():
            messagebox.showerror("ComfyUI", "ComfyUI venv not found. Install ComfyUI first.")
            return
        port = int(comfy_port_var.get())
        # ComfyUI main entrypoint is main.py
        procs["comfyui"] = start_process([str(py), "main.py", "--port", str(port)], cwd=comfy_path, logq=logq)
        webbrowser.open(f"http://127.0.0.1:{port}")

    def stop_all():
        for k, p in list(procs.items()):
            try:
                p.terminate()
            except Exception:
                pass
            procs.pop(k, None)
        log_put(logq, "[stop] Stopped all launched processes (best effort).")

    ttk.Button(launch, text="Start EDMG UI", command=start_edmg_ui, width=18).grid(row=1, column=0, padx=6, pady=6, sticky="w")
    ttk.Button(launch, text="Start A1111", command=start_a1111_ui, width=18).grid(row=1, column=1, padx=6, pady=6, sticky="w")
    ttk.Button(launch, text="Start ComfyUI", command=start_comfyui_ui, width=18).grid(row=1, column=2, padx=6, pady=6, sticky="w")
    ttk.Button(launch, text="Stop All", command=stop_all, width=18).grid(row=2, column=0, padx=6, pady=10, sticky="w")

    root.mainloop()


if __name__ == "__main__":
    main()
