# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# List all your local modules here to ensure they are collected
added_files = [
    # ( 'source_path', 'destination_folder' )
]

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'pywin32',
        'win32gui',
        'win32ui',
        'win32con',
        'mss',
        'cv2'
    ],
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
    name='GeminiStreamProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Set to True if you want a terminal window for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=['assets/icon.ico'], # Uncomment if you have an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GeminiStreamProcessor',
)