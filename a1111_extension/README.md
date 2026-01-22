# Enhanced Deforum Music Generator (Automatic1111 Extension)

This extension adds a **Deforum Music Generator** script to Automatic1111 WebUI (txt2img -> Scripts).
It analyzes an audio file and generates Deforum-compatible settings (JSON) that you can load into the Deforum extension.

## Install
1. Close WebUI.
2. Copy this folder into: `stable-diffusion-webui/extensions/`
3. Start WebUI once so dependencies install (see console).
4. In WebUI: **txt2img -> Scripts -> Deforum Music Generator**

## Workflow
1. Select an audio file.
2. Set base prompt + style.
3. Click **Generate Deforum Settings**.
4. Click **Save JSON to Deforum Settings Folder** (auto-detects Deforum settings directory).
5. In the Deforum extension, use its **Load Settings** button to load the saved JSON.

## Notes
- Whisper is installed with `--no-deps` to avoid torch conflicts (A1111 already provides torch).
- You still need **FFmpeg** on PATH for some audio formats.
