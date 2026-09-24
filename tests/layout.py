"""Where the project's modules live, for the tests that scan all of them.

Three structural tests -- path discipline, shared constants, and the
published figures -- each work by reading every module in the project and
asserting something about all of them. Each used to spell "every module" as
`os.listdir(ROOT)`, which was exactly right while all thirty-one sat in the
root and silently wrong the moment they did not: `listdir` still returns a
list, the loop still runs, and the assertion passes over a set of one.

That is the specific failure this file exists to prevent. The layout is
written down once, and every scan that depends on it asserts it found
something, so a future move produces a failure rather than a smaller truth.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The packages the shipped code is organised into, in dependency order --
# core knows about nothing, pipeline builds on core, analysis reads what the
# pipeline measures, cli and app.py are entry points on top.
PACKAGES = ("core", "pipeline", "analysis", "cli")

# Modules that ship but sit in the root because something has to. app.py is
# the Flask entry point and `python app.py` is a documented command.
ROOT_MODULES = ("app.py",)

# Developer scripts. Shipped in the repository, omitted from coverage, and
# not part of the running system -- but still code, and still bound by the
# rules about resolving paths.
SCRIPT_DIRS = ("tools",)

_SKIP = {"__pycache__", ".git", ".venv", "venv", "node_modules"}


def _walk(folder):
    out = []
    base = os.path.join(ROOT, folder)
    for here, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _SKIP]
        for name in sorted(names):
            if name.endswith(".py"):
                out.append(os.path.relpath(os.path.join(here, name),
                                           ROOT).replace("\\", "/"))
    return sorted(out)


def shipped_modules():
    """Every module that is part of the running system, root-relative.

    Excludes tests/ and tools/: one is where literals belong and the other
    is developer scripting, which is the same split the old
    `not name.startswith("tools")` filter made when everything was flat.
    """
    found = list(ROOT_MODULES)
    for package in PACKAGES:
        found += _walk(package)
    assert found, "no shipped module was found, so this scan is vacuous"
    return sorted(found)


def project_modules():
    """Shipped code plus the developer scripts in tools/."""
    found = shipped_modules()
    for folder in SCRIPT_DIRS:
        found += _walk(folder)
    return sorted(found)


def source_of(name):
    """One module's text, given a root-relative name."""
    with open(os.path.join(ROOT, name.replace("/", os.sep)),
              encoding="utf-8") as handle:
        return handle.read()
