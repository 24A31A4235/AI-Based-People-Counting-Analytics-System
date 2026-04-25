# run.py - Launcher script for People Counting System
# Handles Windows DLL compatibility fixes before importing the app
#
# Usage:
#   python run.py                     # Start with webcam 0
#   python run.py --source 1          # Use webcam 1  
#   python run.py --source video.mp4  # Use video file
#   python run.py --confidence 0.5    # Adjust detection confidence

import os
import sys

# ============================================================
# Fix for Windows + Python 3.14 PyTorch DLL loading issue
# Must happen BEFORE any torch-dependent imports
# ============================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
try:
    _torch_lib = os.path.join(
        os.environ.get('APPDATA', ''), 'Python', 'Python314',
        'site-packages', 'torch', 'lib'
    )
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
except (OSError, AttributeError):
    pass

# Pre-import torch to cache it before any modules try to load it
import torch  # noqa: E402, F401

# Now safely import the main application
from main import PeopleCountingApp, parse_args  # noqa: E402

if __name__ == "__main__":
    args = parse_args()
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    app = PeopleCountingApp(
        source=source,
        model_path=args.model,
        confidence=args.confidence
    )
    sys.exit(app.run())
