"""mitraav-setup: install optional add-ons from the command line. Scans never install or download.

  mitraav-setup                    status, install the small defaults (sounds, scene), finish any
                                   add-on whose package is already installed
  mitraav-setup --list             status only
  mitraav-setup --install birds    install one or more add-ons: sounds scene birds nature
  mitraav-setup --nature           same as --install nature (about 1.7 GB; never installed by default)
  mitraav-setup --plugins          list plugin probe files and whether each is enabled
  mitraav-setup --new-plugin       copy the example plugin into the plugins folder
  mitraav-setup --enable-plugin F  enable a plugin file by name (it is code: read it first)
  mitraav-setup --disable-plugin F
"""
from __future__ import annotations

import sys

from .core import addons

DEFAULTS = ("sounds", "scene")


def _print_status() -> None:
    print("Add-ons:")
    for a in addons.registry():
        print(f"  {a.key:8s} {a.status():14s} {a.name} ({a.size})")
    print("\nSystem tools (install with your package manager):")
    for tool, found, how in addons.system_tools():
        print(f"  {tool:9s} {'found' if found else 'MISSING':8s} {'' if found else how}")


def _plugins(argv: list[str]) -> None:
    import shutil
    from pathlib import Path

    from .core import plugins
    folder = plugins.plugin_dirs()[0]
    if "--new-plugin" in argv:
        folder.mkdir(parents=True, exist_ok=True)
        template = Path(plugins.__file__).parent.parent / "templates" / "example_plugin.py"
        dest = folder / "loudest_moment.py"
        if dest.exists():
            print(f"{dest} already exists")
        else:
            shutil.copy(template, dest)
            print(f"Copied the example plugin to {dest} (it is off until you enable it)")
    for flag, fn in (("--enable-plugin", plugins.enable), ("--disable-plugin", plugins.disable)):
        if flag in argv:
            name = argv[argv.index(flag) + 1] if argv.index(flag) + 1 < len(argv) else ""
            match = [p for p in plugins.list_plugins() if p.path.name == name]
            if not match:
                print(f"No plugin file named '{name}' in {folder}")
            else:
                fn(match[0].path)
                print(f"{name}: {'enabled' if fn is plugins.enable else 'disabled'}")
    print(f"Plugin folder: {folder}")
    for p in plugins.list_plugins():
        err = plugins.ERRORS.get(str(p.path.resolve()), "") or p.error
        print(f"  {p.path.name:28s} {p.status:9s} {p.doc}" + (f"   [{err}]" if err else ""))
    if not plugins.list_plugins():
        print("  (none)")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    print("MitraAV setup (installs and downloads happen only here, never during a scan)\n")
    if any(a in argv for a in ("--plugins", "--new-plugin", "--enable-plugin", "--disable-plugin")):
        _plugins(argv)
        return
    if "--list" in argv:
        _print_status()
        return
    wanted: list[str] = []
    if "--install" in argv:
        wanted = [a for a in argv[argv.index("--install") + 1:] if not a.startswith("--")]
    if "--nature" in argv:
        wanted.append("nature")
    if not wanted:
        wanted = list(DEFAULTS) + [a.key for a in addons.registry()
                                   if a.status() == "partly installed" and a.key not in DEFAULTS
                                   and not a.heavy]
    for key in dict.fromkeys(wanted):
        a = addons.by_key(key)
        if a is None:
            print(f"Unknown add-on '{key}'. Choose from: {', '.join(x.key for x in addons.registry())}")
            continue
        if a.status() == "installed":
            print(f"{a.name}: already installed")
            continue
        print(f"Installing {a.name} ({a.size}) ...")
        try:
            addons.run_steps(a, lambda line: print("  " + line))
        except Exception as e:
            print(f"  FAILED: {e}")
    print()
    _print_status()


if __name__ == "__main__":
    main()
