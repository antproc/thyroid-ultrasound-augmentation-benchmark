from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path

    @classmethod
    def from_args(cls, project_root: str | None = None, data_root: str | None = None) -> "ProjectPaths":
        root = Path(project_root).expanduser().resolve() if project_root else Path.cwd().resolve()
        data = Path(data_root).expanduser().resolve() if data_root else (root / "data").resolve()
        return cls(project_root=root, data_root=data)

    def ensure_output_dirs(self) -> None:
        for rel in [
            "reports",
            "manifests",
            "configs/upstream_snapshot",
            "scan_masks",
            "panel_maps",
            "qc/dataset_audit",
            "qc/scan_mask_overlays",
            "qc/ddti_split_screen",
            "results",
            "figures",
            "logs",
        ]:
            (self.project_root / rel).mkdir(parents=True, exist_ok=True)

def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
