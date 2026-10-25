
# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import glob
import os
import re
import sys


def _matches_runtime(path):
    """Return True when a CARLA binary package matches this CPython runtime."""
    name = os.path.basename(path).lower()
    major, minor = sys.version_info[:2]
    tags = (
        f"cp{major}{minor}",
        f"py{major}.{minor}",
        f"py{major}{minor}",
    )
    # Plain source directories are not ABI-tagged.
    if os.path.isdir(path):
        return True
    tagged = bool(re.search(r"(?:cp|py)\d", name))
    return (not tagged) or any(tag in name for tag in tags)


def _candidate_paths(root):
    patterns = [
        os.path.join(root, "PythonAPI", "carla", "dist", "carla-*.egg"),
        os.path.join(root, "PythonAPI", "carla", "dist", "carla-*.whl"),
        os.path.join(root, "PythonAPI", "carla"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return found


def load_carla(carla_install=None):
    """Import the CARLA client that matches the running CPython ABI.

    CARLA 0.9.13's official Windows binaries target Python 3.6-3.8.  LIVE V1.1
    therefore ships a dedicated Python-3.8 setup script instead of modifying
    the user's normal/offline LVIMOT environment.
    """
    first_error = None
    try:
        import carla  # type: ignore
        return carla
    except Exception as exc:  # DLL/ABI failures are not always ImportError.
        first_error = exc

    discovered = []
    if carla_install:
        root = os.path.abspath(os.path.expanduser(carla_install))
        discovered = _candidate_paths(root)
        for path in discovered:
            if not _matches_runtime(path):
                continue
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                import carla  # type: ignore
                return carla
            except Exception:
                continue

    runtime = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    found_text = ", ".join(os.path.basename(p) for p in discovered) or "none"
    raise ImportError(
        "CARLA Python API could not be imported. "
        f"Current Python is {runtime}; discovered CARLA packages: {found_text}. "
        "CARLA 0.9.13 Windows binaries require CPython 3.6, 3.7 or 3.8. "
        "Run setup_live_carla_env.bat once, then run_lvimot_gui.bat. "
        f"Initial import error: {first_error!r}"
    )
