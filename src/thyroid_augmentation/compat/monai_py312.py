from __future__ import annotations

import importlib
import importlib.machinery
import sys

class _LoadedModuleShim:
    def load_module(self, fullname: str):
        return sys.modules.get(fullname) or importlib.import_module(fullname)

def ensure_monai_py312_compat() -> None:
    if hasattr(importlib.machinery.FileFinder, "find_module"):
        return

    def find_module(self, fullname: str, path: object | None = None):
        return _LoadedModuleShim()

    importlib.machinery.FileFinder.find_module = find_module
