import os
import sys
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.train import main

if __name__ == "__main__":
    main() 