"""
LuminaRetouch - Professional Portrait Retouching & AI Upscaling
Entry point
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main_window import main

if __name__ == "__main__":
    main()
