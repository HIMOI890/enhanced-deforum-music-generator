const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let apiProc = null;

function repoRoot() {
  // desktop/electron/electron/main.js -> repo root
  return path.resolve(__dirname, '..', '..', '..');
}

function pythonExe() {
  // Allow overriding to a venv python
  return process.env.EDMG_PYTHON || 'python';
}

function startApi() {
  if (apiProc) return;
  const root = repoRoot();
  const env = { ...process.env };
  env.PYTHONPATH = env.PYTHONPATH ? `${root}:${env.PYTHONPATH}` : root;
  env.EDMG_API_HOST = env.EDMG_API_HOST || '127.0.0.1';
  env.EDMG_API_PORT = env.EDMG_API_PORT || '7861';

  apiProc = spawn(pythonExe(), ['-m', 'scripts.run_api'], {
    cwd: root,
    env,
    stdio: 'pipe'
  });

  apiProc.stdout.on('data', (d) => {
    process.stdout.write(`[API] ${d}`);
  });
  apiProc.stderr.on('data', (d) => {
    process.stderr.write(`[API] ${d}`);
  });
  apiProc.on('exit', (code) => {
    apiProc = null;
    console.log(`API exited with code ${code}`);
  });
}

function stopApi() {
  if (!apiProc) return;
  try {
    apiProc.kill();
  } catch (_) {}
  apiProc = null;
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  startApi();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopApi();
  if (process.platform !== 'darwin') app.quit();
});
