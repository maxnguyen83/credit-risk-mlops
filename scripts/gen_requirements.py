"""Regenerate requirements*.txt from pyproject.toml.

The submission checklist names requirements.txt, but pyproject.toml is where the
pins actually live -- the package, the Dockerfiles and CI all install from it.
Two files listing the same dependencies is one file too many to keep in sync by
hand, so this derives them and nobody edits the generated ones.

    python scripts/gen_requirements.py
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADER = """# Generated from pyproject.toml -- that file is the source of truth.
# Regenerate with: python scripts/gen_requirements.py
#
# It exists because the DDM501 submission checklist names requirements.txt.
# The Dockerfiles and CI install the package itself (`pip install .`), which
# resolves these same pins plus the entry points -- so this file is a
# convenience for a reader, never the thing that is actually installed.
"""

DEV_HEADER = """# Generated from pyproject.toml [project.optional-dependencies].dev
# Regenerate with: python scripts/gen_requirements.py
"""


def main() -> int:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime = cfg["project"]["dependencies"]
    dev = cfg["project"]["optional-dependencies"]["dev"]

    (ROOT / "requirements.txt").write_text(HEADER + "\n" + "\n".join(runtime) + "\n")
    (ROOT / "requirements-dev.txt").write_text(
        DEV_HEADER + "\n-r requirements.txt\n\n" + "\n".join(dev) + "\n"
    )
    print(f"wrote requirements.txt ({len(runtime)} pins) and requirements-dev.txt ({len(dev)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
