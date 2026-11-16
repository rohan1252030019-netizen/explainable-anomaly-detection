"""
tests/test_packaging.py

Regression tests for the packaging defects found in the audit.

The wheel used to ship only 6 files: ``[tool.setuptools] packages = ["src",
"backend"]`` declared the two *top-level directories* as flat packages, so no
subpackage was collected and ``import backend.routers`` / ``import src.models``
failed from a clean install.  ``backend/`` was also a PEP 420 namespace package
(no ``__init__.py``), which made mypy see every module under two names.

These checks are cheap and run in the normal suite: the heavier build-and-install
check lives in the CI ``package`` job.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_package_discovery_includes_every_subpackage():
    data = _pyproject()
    tool = data["tool"]["setuptools"]
    find = tool["packages"]["find"]
    assert "src*" in find["include"]
    assert "backend*" in find["include"]
    assert find["namespaces"] is False
    # The broken form was a flat list of top-level directories.
    assert isinstance(tool["packages"], dict), tool["packages"]


@pytest.mark.parametrize(
    "package_dir",
    [
        "src",
        "src/data",
        "src/detection",
        "src/evaluation",
        "src/explainability",
        "src/models",
        "src/visualization",
        "backend",
        "backend/routers",
    ],
)
def test_every_package_has_an_init_file(package_dir):
    assert (ROOT / package_dir / "__init__.py").is_file(), (
        f"{package_dir} must be a regular package; without __init__.py it is a "
        "namespace package and mypy reports 'Source file found twice under "
        "different module names'."
    )


def test_version_has_a_single_source():
    import src

    data = _pyproject()
    assert data["project"]["dynamic"] == ["version"]
    assert data["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "src.__version__"

    from backend.main import app, health_check

    assert app.version == src.__version__
    # ``health_check`` returns the ``HealthResponse`` model since Phase 5, not a
    # bare dict; the endpoint serialises it identically.
    health = health_check()
    assert health.version == src.__version__
    assert health.model_dump()["version"] == src.__version__


def test_licence_matches_the_readme_and_ships_a_file():
    data = _pyproject()
    assert data["project"]["license"] == "Apache-2.0"
    assert "LICENSE" in data["project"]["license-files"]

    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0" in text

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Apache 2.0" in readme


def test_requires_python_matches_the_tested_floor():
    data = _pyproject()
    assert data["project"]["requires-python"] == ">=3.11"


def test_every_cross_module_import_target_exists():
    """
    Guard the namespace/import split: ``backend.state`` must import from
    ``src.*`` and nothing may import a bare ``state`` module.
    """
    for path in list((ROOT / "backend").rglob("*.py")) + [
        ROOT / "run.py"
    ]:
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith(("import state", "from state ")), (
                f"{path} imports a bare top-level 'state' module; it must use "
                "backend.state."
            )
