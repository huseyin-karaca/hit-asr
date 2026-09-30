"""Search spaces as data: a `Param` per dimension, frozen or searched, and a space's pasteable source — how a
notebook spells Table 2 out and a record stores it."""

__all__ = ['Param', 'cat', 'space_defaults', 'space_source', 'space_from_source']

from dataclasses import dataclass

import numpy as np


@dataclass
class Param:
    """One searchable dimension, with the physical limits it may never leave.

    `low`/`high` is the current range (`log=True` samples it geometrically),
    `choices` the levels of a categorical, `hard_*` bound any expansion, and
    `frozen` takes the dimension out of the search at a fixed value without
    removing it from the record.
    """
    name: str
    kind: str                      # "float" | "int" | "cat"
    low: float = None
    high: float = None
    log: bool = False
    choices: tuple = ()
    hard_low: float = None
    hard_high: float = None
    frozen: object = None

    def suggest(self, trial):
        if self.frozen is not None:
            return self.frozen
        if self.kind == "cat":
            return trial.suggest_categorical(self.name, list(self.choices))
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)

    def default(self):
        """A representative value: the frozen one, the first choice, or the geometric/arithmetic midpoint."""
        if self.frozen is not None:
            return self.frozen
        if self.kind == "cat":
            return self.choices[0]
        mid = (np.sqrt(self.low * self.high) if self.log else (self.low + self.high) / 2)
        return int(round(mid)) if self.kind == "int" else float(mid)


def cat(name, *values):
    return Param(name, "cat", choices=tuple(values))


def space_defaults(space):
    """`{name: default}` for every parameter of a space — what `build` gets with no search."""
    return {n: p.default() for n, p in space.items()}


# ------------------------------------------------------------------ arms --


def _param_source(p):
    args = [repr(p.name), repr(p.kind)]
    if p.kind == "cat":
        args.append(f"choices={tuple(p.choices)!r}")
    else:
        args += [f"low={p.low!r}", f"high={p.high!r}"]
        if p.log:
            args.append("log=True")
        if p.hard_low is not None:
            args.append(f"hard_low={p.hard_low!r}")
        if p.hard_high is not None:
            args.append(f"hard_high={p.hard_high!r}")
    if p.frozen is not None:
        args.append(f"frozen={p.frozen!r}")
    return f"Param({', '.join(args)})"


def space_source(space, varname="SPACE"):
    """A space as pasteable Python — what the manuscript notebooks spell out and Table 2 reads."""
    lines = [f'SPACES["{varname}"] = {{p.name: p for p in [']
    lines += [f"    {_param_source(p)}," for p in space.values()]
    lines.append("]}")
    return "\n".join(lines)


def space_from_source(src):
    """The inverse of `space_source`: a space from its pasteable source — how the main records store Table 2
    (`search_spaces`), and how `22`/`23` read the good box back. The source is this project's own record."""
    ns = {"SPACES": {}, "Param": Param, "cat": cat}
    exec(src, ns)
    if len(ns["SPACES"]) != 1:
        raise ValueError(f"expected one SPACES[...] assignment, found {len(ns['SPACES'])}")
    return next(iter(ns["SPACES"].values()))
