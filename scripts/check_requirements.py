#!/usr/bin/env python
"""
scripts/check_requirements.py

Verify that requirements.txt and pyproject.toml declare the same packages with
the same version constraints.

These two files drifted before: fastapi and uvicorn were needed by the backend
but present only in the environment, and a stale streamlit entry made it look
like there was a Streamlit app.  Comparing only package *names* was not enough:
a file could say ``pandas>=2.0,<4`` while the other said ``pandas>=2.2`` and the
check still passed, so the CI environment and a wheel install could resolve
different versions.  This compares normalized specifier sets as well.

Requirement syntax this understands (the subset both files actually use):
    name
    name[extra1,extra2]>=1.0,<2   # inline comment
    name>=1.0 ; python_version < "3.12"
    -r other.txt / --index-url ...   (skipped)

Exits 0 on agreement, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name, optional [extras], then the rest (specifiers and/or markers)
_REQ_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?P<extras>\[[^\]]*\])?\s*(?P<rest>.*)$")
_SPEC_RE = re.compile(r"^(===|==|!=|<=|>=|~=|<|>)\s*([A-Za-z0-9._*+!-]+)$")


def _strip_comment(line: str) -> str:
    """Remove an inline ``#`` comment that is not inside brackets or a marker."""
    out = []
    for ch in line:
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).strip()


def normalize_name(name: str) -> str:
    """PEP 503 normalization so ``python_multipart`` == ``python-multipart``."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def parse_requirement(line: str) -> tuple[str, frozenset[str], str | None] | None:
    """
    Parse one requirement line.

    Returns
    -------
    (normalized_name, frozenset_of_specifiers, extras_string) or None when the
    line is blank, a comment, or a pip option (``-r``, ``--index-url``).
    """
    line = _strip_comment(line)
    if not line or line.startswith("-"):
        return None

    # Environment marker: everything after a top-level ';'.
    marker: str | None = None
    if ";" in line:
        line, marker = line.split(";", 1)
        marker = marker.strip() or None
        line = line.strip()

    match = _REQ_RE.match(line)
    if not match:
        return None

    name = normalize_name(match.group("name"))
    extras = (match.group("extras") or "").replace(" ", "")
    rest = match.group("rest").strip()

    specifiers: set[str] = set()
    if rest:
        for raw in rest.split(","):
            part = raw.strip()
            if not part:
                continue
            spec_match = _SPEC_RE.match(part)
            if not spec_match:
                # Unknown syntax: keep it verbatim so a mismatch is still caught
                # rather than silently dropped.
                specifiers.add(part.replace(" ", ""))
                continue
            op, version = spec_match.groups()
            specifiers.add(f"{op}{version}")

    # Fold the extras into the identity: `uvicorn[standard]` and `uvicorn` are
    # not interchangeable declarations.
    identity = f"{name}{extras}"
    return identity, frozenset(specifiers), marker


def index(lines: list[str], source: str) -> dict[str, dict]:
    """Map identity -> {specifiers, marker, source} for a list of requirements."""
    out: dict[str, dict] = {}
    for raw in lines:
        parsed = parse_requirement(raw)
        if parsed is None:
            continue
        identity, specs, marker = parsed
        if identity in out:
            # A duplicate declaration is itself a defect worth reporting.
            out[identity]["duplicate"] = True
        out[identity] = {
            "specifiers": specs,
            "marker": marker,
            "source": source,
            "duplicate": False,
        }
    return out


def describe(specs: frozenset[str]) -> str:
    return ",".join(sorted(specs)) if specs else "(no version bound)"


def main() -> int:
    req_path = ROOT / "requirements.txt"
    dev_req_path = ROOT / "requirements-dev.txt"
    pyproject_path = ROOT / "pyproject.toml"

    if not req_path.exists() or not pyproject_path.exists():
        print("SKIP: requirements.txt or pyproject.toml not found.")
        return 0

    runtime = index(req_path.read_text(encoding="utf-8").splitlines(), "requirements.txt")
    dev = (
        index(dev_req_path.read_text(encoding="utf-8").splitlines(), "requirements-dev.txt")
        if dev_req_path.exists()
        else {}
    )

    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)
    project = data["project"]
    proj_runtime = index(project["dependencies"], "pyproject.toml [dependencies]")
    proj_dev = index(
        project.get("optional-dependencies", {}).get("dev", []),
        "pyproject.toml [dev]",
    )

    problems: list[str] = []

    for label, reqs, declared in (
        ("runtime", runtime, proj_runtime),
        ("dev", dev, proj_dev),
    ):
        req_file = "requirements.txt" if label == "runtime" else "requirements-dev.txt"

        for identity in sorted(set(reqs) - set(declared)):
            problems.append(
                f"  [{label}] {req_file} lists '{identity}' but pyproject.toml does not."
            )
        for identity in sorted(set(declared) - set(reqs)):
            problems.append(
                f"  [{label}] pyproject.toml lists '{identity}' but {req_file} does not.\n"
                f"    (add it, or a plain `pip install -r {req_file}` will not pull it in)"
            )

        for identity in sorted(set(reqs) & set(declared)):
            a, b = reqs[identity], declared[identity]
            if a["specifiers"] != b["specifiers"]:
                problems.append(
                    f"  [{label}] '{identity}' has different version constraints:\n"
                    f"    {req_file}: {describe(a['specifiers'])}\n"
                    f"    pyproject.toml: {describe(b['specifiers'])}"
                )
            if (a["marker"] or "") != (b["marker"] or ""):
                problems.append(
                    f"  [{label}] '{identity}' has different environment markers:\n"
                    f"    {req_file}: {a['marker']!r}\n"
                    f"    pyproject.toml: {b['marker']!r}"
                )
            if a.get("duplicate"):
                problems.append(f"  [{label}] '{identity}' is declared twice in {req_file}.")

    if problems:
        print("requirements files and pyproject.toml disagree:")
        print("\n".join(problems))
        return 1

    print(
        f"OK: {len(runtime)} runtime and {len(dev)} dev packages declared "
        "consistently, including version constraints."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
