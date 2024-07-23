import sys
from pathlib import Path

# In this example, assuming __init__.py is directly in the root directory:
root_dir = str(Path(__name__).resolve().parent)

# Add the root directory to sys.path if it's not already there
if root_dir not in sys.path:
    sys.path.append(root_dir)
