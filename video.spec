# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)
import importlib.util

block_cipher = None

def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

hiddenimports = [
    "win32gui",
    "win32ui",
    "win32con",
    "win32api",
    "pywintypes",
    "pythoncom",
    "mss",
    "cv2",
    "PyQt6.QtMultimedia",
]

# --- yt-dlp python module is yt_dlp (underscore) ---
hiddenimports += collect_submodules("yt_dlp")

# Optional yt_dlp-related deps (only add if installed)
for opt in ["mutagen", "websockets", "brotli", "certifi", "urllib3", "requests"]:
    if _has(opt):
        hiddenimports.append(opt)

datas = []
datas += collect_data_files("yt_dlp", include_py_files=True)

# If certifi is installed, bundle its CA bundle
if _has("certifi"):
    datas += collect_data_files("certifi")

binaries = []
binaries += collect_dynamic_libs("cv2")

a = Analysis(
    ["gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NatesStreamProcessor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NatesStreamProcessor",
)
