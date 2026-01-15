# Enhanced Deforum Music Generator (EDMG) — Unified Repo

This repo merges:
- Standalone EDMG (Gradio UI + CLI + API)
- Automatic1111 extension bundle (included in `/a1111_extension`)
- Installer scripts (root + `installer_gui.py`)

## Quick start (recommended)

### 1) Run the GUI installer
```bash
python installer_gui.py
```

In the installer:
- Choose **Compute Backend**: `cpu` or `cu118/cu121/cu124`
- Select what to install:
  - Standalone EDMG (venv + deps)
  - Automatic1111 + Deforum + EDMG extension
  - ComfyUI

### 2) Launch
From the installer **Launch** tab:
- **Start EDMG UI** (default port 7860)
- **Start A1111** (default port 7861)
- **Start ComfyUI** (default port 8188)

## CLI / scripts (manual)

### Install (creates `./venv`)
Linux/Mac:
```bash
bash install.sh full cpu
# or CUDA (example)
bash install.sh full cu121
```

Windows:
```powershell
.\install.ps1 -Mode full -Cuda
# or use the GUI installer to choose cu118/cu121/cu124
```

### Run EDMG UI
Linux/Mac:
```bash
./start.sh
```

Windows:
```powershell
.\start.bat
```

## UI default mode: Deforum JSON Expert

The Gradio UI defaults to **“Deforum JSON Expert”** mode:
- A full Deforum settings template is shown as editable JSON
- EDMG generates audio-reactive schedules + prompts
- Your edited template **overrides** the generated output keys when merged
- One-click export to a Deforum-ready ZIP package

## Automatic1111 extension bundle

The A1111 extension folder is included at:
- `a1111_extension/`

The installer will copy it into:
- `stable-diffusion-webui/extensions/enhanced-deforum-music-generator/`

## Notes

- This project installs Python dependencies but does **not** install GPU drivers.
- First run of A1111 can take time (it creates its own venv and installs deps).
