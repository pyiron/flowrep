"""Generated names must not collide with live symbols in the emitted Python scope.

The compiler inlines ``for``/``if``/``while``/``try`` bodies into the enclosing ``def``,
because Python's flow control introduces no scope. Recipe port names, however, are scoped
per recipe. Where round-tripping forces the compiler to *pin* a symbol to a port name, a
synthetic name minted in one scope can land on a live symbol of another -- and the emitted
function then computes something different from the one that was parsed.

Note that recipe round-trip equality does *not* catch most of these. The parser does not
model a for-body assignment leaking into the enclosing Python scope, so re-parsing the
broken source yields back the correct recipe -- two compensating errors. Only the
behavioural assertion forces them. Both are kept: recipe equality still catches
``compiler_minted_collision``, and it must not regress.
"""

import unittest

from flowrep import std
from flowrep.compiler import source
from flowrep.parsers import workflow_parser

from flowrep_static import library, makers


@workflow_parser.workflow
def remapped_collection_for(payload, comp):
    """The collection symbol is remapped to the loop variable, hiding it from the
    body's dedup: the body regenerates ``x_0`` and stomps the enclosing ``x_0``."""
    ys = []
    x_0 = payload.xs
    for n in x_0:
        d = library.MyDataclass(comp, n)
        ys.append(d.x)
    return ys, x_0


@workflow_parser.workflow
def for_inside_if(payload, comp):
    """Same collision, one scope deeper -- the ``if`` body is inlined too."""
    x_0 = payload.xs
    if library.is_positive(payload.num):
        ys = []
        for n in x_0:
            d = library.MyDataclass(comp, n)
            ys.append(d.x)
    else:
        ys = []
        for n in x_0:
            d = library.MyDataclass(comp, n)
            ys.append(d.a)
    return ys, x_0


@workflow_parser.workflow
def for_inside_try(payload, comp):
    """Same collision inside a ``try`` body."""
    x_0 = payload.xs
    try:
        ys = []
        for n in x_0:
            d = library.MyDataclass(comp, n)
            ys.append(d.x)
    except ValueError:
        ys = []
        for n in x_0:
            d = library.MyDataclass(comp, n)
            ys.append(d.a)
    return ys, x_0


@workflow_parser.workflow
def sequential_loops(payload, comp):
    """The first loop stomps the collection the second one reads.

    ``x_0`` is never returned, so the pin comes from the for-node's *input* port
    rather than a workflow output.
    """
    x_0 = payload.xs
    ys = []
    for n in x_0:
        d = library.MyDataclass(comp, n)
        ys.append(d.x)
    zs = []
    for m in x_0:
        e = std.identity(m)
        zs.append(e)
    return ys, zs


@workflow_parser.workflow
def compiler_minted_collision(comp, seed):
    """No attribute access in any body -- the collision is the *compiler's* doing.

    ``a`` and ``b`` feed only an ordinary node, so the compiler names them freely: two
    ``val`` nodes make the allocator mint ``val`` and ``val_0``. The condition's
    attribute chain then generates the input port ``val_0``, pinning its getattr to that
    same symbol -- overwriting ``b`` before ``my_add`` reads it.

    Nothing here collides at *parse* time, so no amount of parser-side deduplication can
    prevent this; only the compiler reserving its pins can.
    """
    a = library.val(1)
    b = library.val(2)
    if library.is_positive(comp.val):  # noqa: SIM108
        m = std.identity(seed)
    else:
        m = library.negate(seed)
    c = std.add(a, b)
    return m, c


class NamespaceHygieneMixin:
    """Assert parse -> emit -> re-parse is a fixed point, and preserves behaviour."""

    workflow: staticmethod
    args: tuple

    def test_round_trips_to_an_equal_recipe(self):
        free = makers.reference_free(self.workflow)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe),
            makers.dump_no_refs(free),
            msg=f"Recipe changed under round trip. Emitted:\n{rendered.source}",
        )

    def test_compiled_source_computes_the_same_function(self):
        free = makers.reference_free(self.workflow)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            fn(*self.args),
            self.workflow(*self.args),
            msg=f"Emitted source is a different function:\n{rendered.source}",
        )


def _payload():
    return library.Payload(xs=[1, 2, 3], num=5, den=2)


def _comp():
    return library.ComplexData(val=7)


class TestRemappedCollectionFor(NamespaceHygieneMixin, unittest.TestCase):
    workflow = staticmethod(remapped_collection_for)

    @property
    def args(self):
        return (_payload(), _comp())


class TestForInsideIf(NamespaceHygieneMixin, unittest.TestCase):
    workflow = staticmethod(for_inside_if)

    @property
    def args(self):
        return (_payload(), _comp())


class TestForInsideTry(NamespaceHygieneMixin, unittest.TestCase):
    workflow = staticmethod(for_inside_try)

    @property
    def args(self):
        return (_payload(), _comp())


class TestSequentialLoops(NamespaceHygieneMixin, unittest.TestCase):
    workflow = staticmethod(sequential_loops)

    @property
    def args(self):
        return (_payload(), _comp())


class TestCompilerMintedCollision(NamespaceHygieneMixin, unittest.TestCase):
    workflow = staticmethod(compiler_minted_collision)

    @property
    def args(self):
        return (_comp(), 4)


if __name__ == "__main__":
    unittest.main()
