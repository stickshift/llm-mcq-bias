from pathlib import Path

import pytest

__all__ = [
    "project_path",
    "datasets_path",
]


@pytest.fixture
def project_path(pytestconfig) -> Path:
    return pytestconfig.rootpath


@pytest.fixture
def datasets_path(project_path: Path) -> Path:
    return project_path / ".build" / "datasets"
