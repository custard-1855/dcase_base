"""pytest のルート設定.

このファイルがレシピ直下にあることで, pytest がこのディレクトリを sys.path に追加し,
`train_pretrained` / `desed_task` / `local` / `sebbs` (submodule) を import できる.
"""

import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent

if str(RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(RECIPE_DIR))
