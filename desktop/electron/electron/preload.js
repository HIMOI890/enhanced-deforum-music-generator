const { contextBridge } = require('electron');

// Expose minimal config to the renderer.
contextBridge.exposeInMainWorld('EDMG', {
  apiBase: () => {
    const host = process.env.EDMG_API_HOST || '127.0.0.1';
    const port = process.env.EDMG_API_PORT || '7861';
    return `http://${host}:${port}`;
  },
});
