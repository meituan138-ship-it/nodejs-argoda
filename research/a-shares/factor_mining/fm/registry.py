"""@factor decorator: every factor declares a name and an economic rationale."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Factor:
    name: str
    rationale: str
    source: str
    func: Callable
    file: str

    @property
    def code_hash(self) -> str:
        """Hash of the formula body (not the name/docs): the same formula re-submitted
        under a new name is the same trial; any change to the body is a new trial."""
        body = self.code.split("\n", 1)[1] if "\n" in self.code else self.code
        return hashlib.sha1("".join(body.split()).encode()).hexdigest()[:12]

    @property
    def code(self) -> str:
        """Function body only (decorator text excluded) — used by the static leak check."""
        src = inspect.getsource(self.func)
        return src[src.index("def "):]


_REGISTRY: list[Factor] = []


def factor(name: str, rationale: str, source: str):
    """Register a factor.

    rationale: WHY it should predict returns (mechanism, who is on the other side).
    source:    where the idea comes from — a paper, a broker report, an Alpha101/191
               formula number, a named market rule (T+1, price limits...), or
               "gp:<run id>" for genetic-programming output. AI must not invent
               factors from nothing; ideas need a traceable origin."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"factor name must be alphanumeric/underscore: {name!r}")
    if len(rationale.strip()) < 10:
        raise ValueError(f"{name}: give a real economic rationale (>= 10 characters)")
    if len(source.strip()) < 4:
        raise ValueError(f"{name}: give the idea's source (paper / report / Alpha191#nn / market rule / gp:run)")

    def deco(fn):
        _REGISTRY.append(Factor(name, rationale.strip(), source.strip(), fn, inspect.getsourcefile(fn) or "?"))
        return fn
    return deco


def load_factor_file(path: str | Path) -> list[Factor]:
    path = Path(path)
    before = len(_REGISTRY)
    spec = importlib.util.spec_from_file_location(f"factors_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return _REGISTRY[before:]
