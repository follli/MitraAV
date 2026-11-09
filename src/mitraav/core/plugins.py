"""User plugin probes: drop a Python file in the plugins folder, review it, and enable it.

A plugin is one `.py` file that defines one or more `Probe` subclasses (see docs/PLUGINS.md). Because
a plugin is arbitrary code that runs with your permissions, nothing loads by itself:

* files are only *listed* (their text is read, never executed) until you enable them;
* enabling records the file's SHA-256, and a file that changes afterwards is disabled again until you
  re-enable it, so an edited or swapped file never runs silently;
* each plugin probe runs in its own process like every other probe, so a crash or hang is contained
  (but a plugin can still do anything your user account can do: read only code you trust).
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .models import model_dir
from .probe import Probe

API_VERSION = 1


def plugin_dirs() -> list[Path]:
    """Folders searched for plugins: MITRAAV_PLUGINS (os.pathsep separated) or the user data folder."""
    env = os.environ.get("MITRAAV_PLUGINS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return [model_dir().parent / "plugins"]


def state_path() -> Path:
    env = os.environ.get("MITRAAV_PLUGIN_STATE")
    return Path(env) if env else model_dir().parent / "plugins.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_state() -> dict[str, str]:
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.get("enabled", {}).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def _save_state(enabled: dict[str, str]) -> None:
    sp = state_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"enabled": enabled}, indent=2), encoding="utf-8")


@dataclass
class PluginFile:
    path: Path
    sha: str
    status: str                 # "enabled" | "disabled" | "changed" (enabled earlier, file edited since)
    doc: str = ""
    probes: list[str] = field(default_factory=list)      # class names found by reading, not running
    error: str = ""

    @property
    def enabled(self) -> bool:
        return self.status == "enabled"


def _describe(path: Path) -> tuple[str, list[str], str]:
    """(first docstring line, probe-looking class names, error) from the source text alone."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as e:
        return "", [], f"cannot be read: {e}"
    doc = (ast.get_docstring(tree) or "").strip().splitlines()
    names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)
             and any(getattr(b, "id", getattr(b, "attr", "")) == "Probe" for b in n.bases)]
    return (doc[0] if doc else ""), names, ""


def list_plugins() -> list[PluginFile]:
    """Every .py file in the plugin folders, with whether it is currently enabled."""
    state = _load_state()
    out = []
    for d in plugin_dirs():
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.py")):
            if p.name.startswith("_"):
                continue
            try:
                sha = _digest(p)
            except OSError:
                continue
            key = str(p.resolve())
            status = "disabled" if key not in state else ("enabled" if state[key] == sha else "changed")
            doc, names, err = _describe(p)
            out.append(PluginFile(p, sha, status, doc, names, err))
    return out


def enable(path: Path) -> None:
    p = Path(path).resolve()
    state = _load_state()
    state[str(p)] = _digest(p)
    _save_state(state)
    ERRORS.pop(str(p), None)


def disable(path: Path) -> None:
    state = _load_state()
    state.pop(str(Path(path).resolve()), None)
    _save_state(state)


ERRORS: dict[str, str] = {}          # plugin path -> why it could not be loaded (shown in the dialog)


def _import(path: Path):
    name = "mitraav_plugin_" + "".join(c if c.isalnum() else "_" for c in path.stem)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # dataclasses and pickling inside the plugin need this
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def classes_in(path: Path) -> list[type[Probe]]:
    """Import one plugin file and return its probe classes (raises on a broken plugin)."""
    mod = _import(path)
    api = getattr(mod, "MITRAAV_API", API_VERSION)
    if not isinstance(api, int) or api > API_VERSION:
        raise RuntimeError(f"needs plugin API {api}, this MitraAV provides {API_VERSION}")
    found = []
    for obj in vars(mod).values():
        if (inspect.isclass(obj) and issubclass(obj, Probe) and obj is not Probe
                and obj.__module__ == mod.__name__ and not inspect.isabstract(obj)):
            if not isinstance(getattr(obj, "name", None), str) or obj.name in ("", "unnamed"):
                raise RuntimeError(f"{obj.__name__} needs a unique `name`")
            obj._plugin_file = str(path.resolve())
            found.append(obj)
    return found


def load_enabled(builtin_names: set[str] | None = None) -> list[type[Probe]]:
    """Probe classes from every enabled plugin whose file is unchanged since it was enabled."""
    taken = set(builtin_names or ())
    out: list[type[Probe]] = []
    for pf in list_plugins():
        key = str(pf.path.resolve())
        if not pf.enabled:
            if pf.status == "changed":
                ERRORS[key] = "the file changed since you enabled it: review it and enable it again"
            continue
        try:
            clash = ""
            for cls in classes_in(pf.path):
                if cls.name in taken:
                    clash = f"probe name '{cls.name}' is already used"
                    continue
                taken.add(cls.name)
                out.append(cls)
            if clash:
                ERRORS[key] = clash
            else:
                ERRORS.pop(key, None)
        except BaseException as e:            # a broken plugin must never break the app
            ERRORS[key] = f"{type(e).__name__}: {e}"
    return out


def load_class(file: str, qualname: str) -> type[Probe]:
    """Used inside a probe's own process: re-import the plugin file and find the class."""
    for cls in classes_in(Path(file)):
        if cls.__qualname__ == qualname:
            return cls
    raise RuntimeError(f"{qualname} not found in {file}")
