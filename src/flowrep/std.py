"""
A standard library of flowrep labeled recipes built around fundamental python
operations and most common use cases.
"""

import operator
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from flowrep.parsers import atomic_parser


@atomic_parser.atomic("absolute")
def abs(a, /):
    return operator.abs(a)


@atomic_parser.atomic("added")
def add(a, b, /):
    return operator.add(a, b)


@atomic_parser.atomic("index")
def index(a, /):
    return operator.index(a)


@atomic_parser.atomic("inverted")
def inv(a, /):
    return operator.inv(a)


@atomic_parser.atomic("inverted")
def invert(a, /):
    return operator.invert(a)


@atomic_parser.atomic("negative")
def neg(a, /):
    return operator.neg(a)


@atomic_parser.atomic("positive")
def pos(a, /):
    return operator.pos(a)


@atomic_parser.atomic("negated")
def not_(a, /):
    return operator.not_(a)


@atomic_parser.atomic("truth")
def truth(a, /):
    return operator.truth(a)


@atomic_parser.atomic("length")
def length_hint(obj, /):
    return operator.length_hint(obj)


@atomic_parser.atomic("difference")
def sub(a, b, /):
    return operator.sub(a, b)


@atomic_parser.atomic("difference")
def isub(a, b, /):
    return operator.isub(a, b)


@atomic_parser.atomic("added")
def iadd(a, b, /):
    return operator.iadd(a, b)


@atomic_parser.atomic("product")
def mul(a, b, /):
    return operator.mul(a, b)


@atomic_parser.atomic("product")
def imul(a, b, /):
    return operator.imul(a, b)


@atomic_parser.atomic("quotient")
def floordiv(a, b, /):
    return operator.floordiv(a, b)


@atomic_parser.atomic("quotient")
def ifloordiv(a, b, /):
    return operator.ifloordiv(a, b)


@atomic_parser.atomic("quotient")
def truediv(a, b, /):
    return operator.truediv(a, b)


@atomic_parser.atomic("quotient")
def itruediv(a, b, /):
    return operator.itruediv(a, b)


@atomic_parser.atomic("remainder")
def mod(a, b, /):
    return operator.mod(a, b)


@atomic_parser.atomic("remainder")
def imod(a, b, /):
    return operator.imod(a, b)


@atomic_parser.atomic("power")
def pow(a, b, /):
    return operator.pow(a, b)


@atomic_parser.atomic("power")
def ipow(a, b, /):
    return operator.ipow(a, b)


@atomic_parser.atomic("conjunction")
def and_(a, b, /):
    return operator.and_(a, b)


@atomic_parser.atomic("conjunction")
def iand(a, b, /):
    return operator.iand(a, b)


@atomic_parser.atomic("disjunction")
def or_(a, b, /):
    return operator.or_(a, b)


@atomic_parser.atomic("disjunction")
def ior(a, b, /):
    return operator.ior(a, b)


@atomic_parser.atomic("exclusive_or")
def xor(a, b, /):
    return operator.xor(a, b)


@atomic_parser.atomic("exclusive_or")
def ixor(a, b, /):
    return operator.ixor(a, b)


@atomic_parser.atomic("left_shifted")
def lshift(a, b, /):
    return operator.lshift(a, b)


@atomic_parser.atomic("left_shifted")
def ilshift(a, b, /):
    return operator.ilshift(a, b)


@atomic_parser.atomic("right_shifted")
def rshift(a, b, /):
    return operator.rshift(a, b)


@atomic_parser.atomic("right_shifted")
def irshift(a, b, /):
    return operator.irshift(a, b)


@atomic_parser.atomic("matrix_product")
def matmul(a, b, /):
    return operator.matmul(a, b)


@atomic_parser.atomic("matrix_product")
def imatmul(a, b, /):
    return operator.imatmul(a, b)


@atomic_parser.atomic("equal")
def eq(a, b, /):
    return operator.eq(a, b)


@atomic_parser.atomic("not_equal")
def ne(a, b, /):
    return operator.ne(a, b)


@atomic_parser.atomic("less")
def lt(a, b, /):
    return operator.lt(a, b)


@atomic_parser.atomic("less_equal")
def le(a, b, /):
    return operator.le(a, b)


@atomic_parser.atomic("greater")
def gt(a, b, /):
    return operator.gt(a, b)


@atomic_parser.atomic("greater_equal")
def ge(a, b, /):
    return operator.ge(a, b)


@atomic_parser.atomic("identical")
def is_(a, b, /):
    return operator.is_(a, b)


@atomic_parser.atomic("not_identical")
def is_not(a, b, /):
    return operator.is_not(a, b)


@atomic_parser.atomic("contains")
def contains(a, b, /):
    return operator.contains(a, b)


@atomic_parser.atomic("count")
def countOf(a, b, /):
    return operator.countOf(a, b)


@atomic_parser.atomic("index")
def indexOf(a, b, /):
    return operator.indexOf(a, b)


@atomic_parser.atomic("concatenated")
def concat(a, b, /):
    return operator.concat(a, b)


@atomic_parser.atomic("concatenated")
def iconcat(a, b, /):
    return operator.iconcat(a, b)


@atomic_parser.atomic("item")
def getitem(a, b, /):
    return operator.getitem(a, b)


@atomic_parser.atomic("set")
def setitem(a, b, c, /):
    return operator.setitem(a, b, c)


@atomic_parser.atomic("deleted")
def delitem(a, b, /):
    return operator.delitem(a, b)


@atomic_parser.atomic("result")
def call(
    obj: Callable,
    /,
    *,
    args_: Iterable[Any] | None = None,
    kwargs_: Mapping[str, Any] | None = None,
):
    args_ = () if args_ is None else args_
    kwargs_ = {} if kwargs_ is None else kwargs_
    return obj(*args_, **kwargs_)


@atomic_parser.atomic("attr")
def get_attr(obj: object, name: str, /):
    """
    getattr doesn't have an inspectable signature until python 3.13, so provide a
    well-formed wrapper.
    """
    return getattr(obj, name)


@atomic_parser.atomic
def identity(x):
    return x
