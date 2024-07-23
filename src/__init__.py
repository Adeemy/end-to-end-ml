import sys
from pathlib import Path

# To import modules from the parent directory in Azure compute cluster
root_dir = Path(__name__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
