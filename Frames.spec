# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Frames.app
Build: .venv/bin/pyinstaller Frames.spec --clean
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

hidden_imports = [
    # uvicorn internals
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # fastapi / starlette
    "fastapi",
    "starlette",
    "starlette.routing",
    "starlette.responses",
    "starlette.staticfiles",
    "starlette.middleware",
    "anyio",
    "anyio._backends._asyncio",
    # imaging / RAW
    "rawpy",
    "cv2",
    "PIL",
    "PIL.Image",
    "numpy",
    # pywebview backends (macOS uses cocoa)
    "webview.platforms.cocoa",
]

datas = [
    ("frontend", "frontend"),
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["streamlit", "tkinter", "matplotlib", "scipy"],
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
    name="Frames",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Frames",
)

app = BUNDLE(
    coll,
    name="Frames.app",
    icon="frames_icon.icns",
    bundle_identifier="com.frames.app",
    info_plist={
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "12.0",
        "NSCameraUsageDescription": "Frames needs camera access for RAW file processing.",
        "NSPhotoLibraryUsageDescription": "Frames accesses your photos folder.",
        "CFBundleDisplayName": "Frames",
    },
)
