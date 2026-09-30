"""Run instances of every concrete retrospective node data type, for storage tests."""

from flowrep import wfms, workflow
from flowrep.retrospective import datastructures

from flowrep_static import library


@workflow
def looped(xs):
    ys = []
    for x in xs:
        y = library.increment(x)
        ys.append(y)
    return ys


@workflow
def branched(n):
    # A statement, not a ternary: only an if-block parses into IfData
    if library.is_positive(n):  # noqa: SIM108
        m = library.increment(n)
    else:
        m = library.decrement(n)
    return m


@workflow
def counted(n):
    while library.is_positive(n):
        n = library.decrement(n)
    return n


@workflow
def guarded(x, y):
    try:
        z = library.raises_custom(x, y)
    except library.MyCustomException:
        z = library.increment(x)
    return z


@workflow
def with_constant(a):
    c = library.increment(a, 5)
    return c


_RUNS = (
    (looped, {"xs": [1, 2]}),
    (branched, {"n": 1}),
    (counted, {"n": 2}),
    (guarded, {"x": 1, "y": 2}),
    (with_constant, {"a": 1}),
)


def samples() -> dict[type[datastructures.NodeData], datastructures.NodeData]:
    """One freshly run instance of each node data type found in these workflows."""
    found: dict[type[datastructures.NodeData], datastructures.NodeData] = {}
    for wf, kwargs in _RUNS:
        data = wfms.run_recipe(wf.flowrep_recipe, **kwargs)
        found.setdefault(type(data), data)
        for child in data.nodes.values():
            found.setdefault(type(child), child)
    return found
