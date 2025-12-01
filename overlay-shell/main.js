const { app, BrowserWindow, globalShortcut } = require('electron');
const path = require('path');

const OVERLAY_URL = process.env.OVERLAY_URL || 'http://127.0.0.1:8001/overlay';
const TOGGLE_KEY = process.env.OVERLAY_TOGGLE_KEY || 'F8';
const FRONT_KEY = process.env.OVERLAY_FRONT_KEY || 'Shift+F8';

function createWindow() {
  const win = new BrowserWindow({
    width: 420,
    height: 640,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: true,
    skipTaskbar: true,
    backgroundColor: '#00000000',
    opacity: parseFloat(process.env.OVERLAY_OPACITY || '0.9'),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const indexPath = path.join(__dirname, 'index.html');
  const urlWithQuery = `file://${indexPath}?url=${encodeURIComponent(OVERLAY_URL)}`;
  win.loadURL(urlWithQuery);
  // Start with click-through off so the window is draggable by default.
  let ignoring = false;
  win.setIgnoreMouseEvents(ignoring, { forward: true });

  globalShortcut.register(TOGGLE_KEY, () => {
    ignoring = !ignoring;
    win.setIgnoreMouseEvents(ignoring, { forward: true });
  });

  globalShortcut.register(FRONT_KEY, () => {
    // Briefly ensure topmost and disable click-through for repositioning.
    win.setAlwaysOnTop(true, 'screen-saver');
    ignoring = false;
    win.setIgnoreMouseEvents(ignoring, { forward: true });
    win.focus();
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});
