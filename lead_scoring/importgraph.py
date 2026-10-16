"""Static discovery of the first-party modules an entry point depends on.

Used by the Docker image test. Walking the import graph with ``ast`` rather
than importing anything means the answer is the same whether or not the
dependencies are installed, and it cannot be fooled by a module that happens
to already be in ``sys.modules``.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _module_to_path(root: Path, module: str) -> Path | None:
    """Resolve a dotted module name to a file inside ``root``, if it is ours."""
    parts = module.split(".")
    candidate = root.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    package_init = root.joinpath(*parts, "__init__.py")
    if package_init.is_file():
        return package_init
    return None


def _imports_in(path: Path, package: str | None) -> set[str]:
    """Dotted names imported by one file, with relative imports resolved."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and package:
                # Relative import: climb `level - 1` packages from the current one.
                base_parts = package.split(".")
                base = ".".join(base_parts[: len(base_parts) - (node.level - 1)] or base_parts)
                prefix = f"{base}.{node.module}" if node.module else base
            elif node.module:
                prefix = node.module
            else:
                continue
            found.add(prefix)
            # `from pkg import mod` may name a submodule rather than an object.
            found.update(f"{prefix}.{alias.name}" for alias in node.names)
    return found


def first_party_modules(entry_module: str, root: Path) -> set[str]:
    """Every module under ``root`` reachable from ``entry_module``, inclusive.

    Third-party and stdlib imports are ignored: they are the dependency
    manifest's job, not this function's.
    """
    start = _module_to_path(root, entry_module)
    if start is None:
        raise ValueError(f"{entry_module} is not a module under {root}")

    seen: set[str] = set()
    queue = [entry_module]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        path = _module_to_path(root, module)
        if path is None:
            continue
        seen.add(module)

        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]

        # Importing a submodule imports its parent packages too, so their
        # __init__.py files have to be in the image as well. Without this the
        # walk would miss exactly the kind of omission it exists to catch.
        parts = module.split(".")
        for depth in range(1, len(parts)):
            parent = ".".join(parts[:depth])
            if parent not in seen and _module_to_path(root, parent) is not None:
                queue.append(parent)

        for imported in _imports_in(path, package):
            if imported not in seen and _module_to_path(root, imported) is not None:
                queue.append(imported)
    return seen
