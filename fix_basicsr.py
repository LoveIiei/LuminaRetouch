"""
Fix basicsr compatibility with newer torchvision versions.
Run this once after installing dependencies:
    python fix_basicsr.py
"""

import site
import sys
from pathlib import Path


def find_basicsr_degradations():
    """Find the basicsr degradations.py file in site-packages."""
    # Check common locations
    search_paths = site.getsitepackages() + [site.getusersitepackages()]

    for base_path in search_paths:
        if base_path is None:
            continue
        degradations_path = Path(base_path) / "basicsr" / "data" / "degradations.py"
        if degradations_path.exists():
            return degradations_path

    # Also check in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = Path(sys.prefix) / "Lib" / "site-packages" / "basicsr" / "data" / "degradations.py"
        if venv_path.exists():
            return venv_path

    return None


def fix_basicsr():
    """
    Fix the torchvision.transforms.functional_tensor import error.

    In torchvision >= 0.15, functional_tensor was renamed to _functional_tensor
    and eventually the rgb_to_grayscale function moved to functional.
    """
    degradations_path = find_basicsr_degradations()

    if degradations_path is None:
        print("Could not find basicsr/data/degradations.py")
        print("Make sure basicsr is installed: pip install basicsr")
        return False

    print(f"Found: {degradations_path}")

    # Read the file
    content = degradations_path.read_text(encoding='utf-8')

    # Check if fix is needed
    old_import = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    new_import = "from torchvision.transforms.functional import rgb_to_grayscale"

    if old_import not in content:
        if new_import in content:
            print("File already patched!")
            return True
        else:
            print("Unexpected file content - import line not found")
            return False

    # Apply fix
    content = content.replace(old_import, new_import)

    # Write back
    try:
        degradations_path.write_text(content, encoding='utf-8')
        print(f"Successfully patched {degradations_path}")
        print(f"Changed: {old_import}")
        print(f"To:      {new_import}")
        return True
    except PermissionError:
        print(f"Permission denied. Try running as administrator or use:")
        print(f'  python -c "exec(open(\'fix_basicsr.py\').read())"')
        return False


if __name__ == "__main__":
    success = fix_basicsr()
    sys.exit(0 if success else 1)
