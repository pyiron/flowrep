"""
A standard library of flowrep labeled recipes built around fundamental python
operations and most common use cases.
"""

import operator
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from flowrep.parsers import atomic_parser
from flowrep.prospective import atomic_recipe


@atomic_parser.atomic("absolute", unpack_mode=atomic_recipe.UnpackMode.NONE)
def abs(a, /):
    return operator.abs(a)


@atomic_parser.atomic("added", unpack_mode=atomic_recipe.UnpackMode.NONE)
def add(a, b, /):
    return operator.add(a, b)


@atomic_parser.atomic("index", unpack_mode=atomic_recipe.UnpackMode.NONE)
def index(a, /):
    return operator.index(a)


@atomic_parser.atomic("inverted", unpack_mode=atomic_recipe.UnpackMode.NONE)
def inv(a, /):
    return operator.inv(a)


@atomic_parser.atomic("inverted", unpack_mode=atomic_recipe.UnpackMode.NONE)
def invert(a, /):
    return operator.invert(a)


@atomic_parser.atomic("negative", unpack_mode=atomic_recipe.UnpackMode.NONE)
def neg(a, /):
    return operator.neg(a)


@atomic_parser.atomic("positive", unpack_mode=atomic_recipe.UnpackMode.NONE)
def pos(a, /):
    return operator.pos(a)


@atomic_parser.atomic("negated", unpack_mode=atomic_recipe.UnpackMode.NONE)
def not_(a, /):
    return operator.not_(a)


@atomic_parser.atomic("truth", unpack_mode=atomic_recipe.UnpackMode.NONE)
def truth(a, /):
    return operator.truth(a)


@atomic_parser.atomic("length", unpack_mode=atomic_recipe.UnpackMode.NONE)
def length_hint(obj, /):
    return operator.length_hint(obj)


@atomic_parser.atomic("difference", unpack_mode=atomic_recipe.UnpackMode.NONE)
def sub(a, b, /):
    return operator.sub(a, b)


@atomic_parser.atomic("difference", unpack_mode=atomic_recipe.UnpackMode.NONE)
def isub(a, b, /):
    return operator.isub(a, b)


@atomic_parser.atomic("added", unpack_mode=atomic_recipe.UnpackMode.NONE)
def iadd(a, b, /):
    return operator.iadd(a, b)


@atomic_parser.atomic("product", unpack_mode=atomic_recipe.UnpackMode.NONE)
def mul(a, b, /):
    return operator.mul(a, b)


@atomic_parser.atomic("product", unpack_mode=atomic_recipe.UnpackMode.NONE)
def imul(a, b, /):
    return operator.imul(a, b)


@atomic_parser.atomic("quotient", unpack_mode=atomic_recipe.UnpackMode.NONE)
def floordiv(a, b, /):
    return operator.floordiv(a, b)


@atomic_parser.atomic("quotient", unpack_mode=atomic_recipe.UnpackMode.NONE)
def ifloordiv(a, b, /):
    return operator.ifloordiv(a, b)


@atomic_parser.atomic("quotient", unpack_mode=atomic_recipe.UnpackMode.NONE)
def truediv(a, b, /):
    return operator.truediv(a, b)


@atomic_parser.atomic("quotient", unpack_mode=atomic_recipe.UnpackMode.NONE)
def itruediv(a, b, /):
    return operator.itruediv(a, b)


@atomic_parser.atomic("remainder", unpack_mode=atomic_recipe.UnpackMode.NONE)
def mod(a, b, /):
    return operator.mod(a, b)


@atomic_parser.atomic("remainder", unpack_mode=atomic_recipe.UnpackMode.NONE)
def imod(a, b, /):
    return operator.imod(a, b)


@atomic_parser.atomic("power", unpack_mode=atomic_recipe.UnpackMode.NONE)
def pow(a, b, /):
    return operator.pow(a, b)


@atomic_parser.atomic("power", unpack_mode=atomic_recipe.UnpackMode.NONE)
def ipow(a, b, /):
    return operator.ipow(a, b)


@atomic_parser.atomic("conjunction", unpack_mode=atomic_recipe.UnpackMode.NONE)
def and_(a, b, /):
    return operator.and_(a, b)


@atomic_parser.atomic("conjunction", unpack_mode=atomic_recipe.UnpackMode.NONE)
def iand(a, b, /):
    return operator.iand(a, b)


@atomic_parser.atomic("disjunction", unpack_mode=atomic_recipe.UnpackMode.NONE)
def or_(a, b, /):
    return operator.or_(a, b)


@atomic_parser.atomic("disjunction", unpack_mode=atomic_recipe.UnpackMode.NONE)
def ior(a, b, /):
    return operator.ior(a, b)


@atomic_parser.atomic("exclusive_or", unpack_mode=atomic_recipe.UnpackMode.NONE)
def xor(a, b, /):
    return operator.xor(a, b)


@atomic_parser.atomic("exclusive_or", unpack_mode=atomic_recipe.UnpackMode.NONE)
def ixor(a, b, /):
    return operator.ixor(a, b)


@atomic_parser.atomic("left_shifted", unpack_mode=atomic_recipe.UnpackMode.NONE)
def lshift(a, b, /):
    return operator.lshift(a, b)


@atomic_parser.atomic("left_shifted", unpack_mode=atomic_recipe.UnpackMode.NONE)
def ilshift(a, b, /):
    return operator.ilshift(a, b)


@atomic_parser.atomic("right_shifted", unpack_mode=atomic_recipe.UnpackMode.NONE)
def rshift(a, b, /):
    return operator.rshift(a, b)


@atomic_parser.atomic("right_shifted", unpack_mode=atomic_recipe.UnpackMode.NONE)
def irshift(a, b, /):
    return operator.irshift(a, b)


@atomic_parser.atomic("matrix_product", unpack_mode=atomic_recipe.UnpackMode.NONE)
def matmul(a, b, /):
    return operator.matmul(a, b)


@atomic_parser.atomic("matrix_product", unpack_mode=atomic_recipe.UnpackMode.NONE)
def imatmul(a, b, /):
    return operator.imatmul(a, b)


@atomic_parser.atomic("equal", unpack_mode=atomic_recipe.UnpackMode.NONE)
def eq(a, b, /):
    return operator.eq(a, b)


@atomic_parser.atomic("not_equal", unpack_mode=atomic_recipe.UnpackMode.NONE)
def ne(a, b, /):
    return operator.ne(a, b)


@atomic_parser.atomic("less", unpack_mode=atomic_recipe.UnpackMode.NONE)
def lt(a, b, /):
    return operator.lt(a, b)


@atomic_parser.atomic("less_equal", unpack_mode=atomic_recipe.UnpackMode.NONE)
def le(a, b, /):
    return operator.le(a, b)


@atomic_parser.atomic("greater", unpack_mode=atomic_recipe.UnpackMode.NONE)
def gt(a, b, /):
    return operator.gt(a, b)


@atomic_parser.atomic("greater_equal", unpack_mode=atomic_recipe.UnpackMode.NONE)
def ge(a, b, /):
    return operator.ge(a, b)


@atomic_parser.atomic("identical", unpack_mode=atomic_recipe.UnpackMode.NONE)
def is_(a, b, /):
    return operator.is_(a, b)


@atomic_parser.atomic("not_identical", unpack_mode=atomic_recipe.UnpackMode.NONE)
def is_not(a, b, /):
    return operator.is_not(a, b)


@atomic_parser.atomic("contains", unpack_mode=atomic_recipe.UnpackMode.NONE)
def contains(a, b, /):
    return operator.contains(a, b)


@atomic_parser.atomic("count", unpack_mode=atomic_recipe.UnpackMode.NONE)
def countOf(a, b, /):
    return operator.countOf(a, b)


@atomic_parser.atomic("index", unpack_mode=atomic_recipe.UnpackMode.NONE)
def indexOf(a, b, /):
    return operator.indexOf(a, b)


@atomic_parser.atomic("concatenated", unpack_mode=atomic_recipe.UnpackMode.NONE)
def concat(a, b, /):
    return operator.concat(a, b)


@atomic_parser.atomic("concatenated", unpack_mode=atomic_recipe.UnpackMode.NONE)
def iconcat(a, b, /):
    return operator.iconcat(a, b)


@atomic_parser.atomic("item", unpack_mode=atomic_recipe.UnpackMode.NONE)
def getitem(a, b, /):
    return operator.getitem(a, b)


@atomic_parser.atomic("set", unpack_mode=atomic_recipe.UnpackMode.NONE)
def setitem(a, b, c, /):
    return operator.setitem(a, b, c)


@atomic_parser.atomic("deleted", unpack_mode=atomic_recipe.UnpackMode.NONE)
def delitem(a, b, /):
    return operator.delitem(a, b)


@atomic_parser.atomic("result", unpack_mode=atomic_recipe.UnpackMode.NONE)
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


@atomic_parser.atomic("attr", unpack_mode=atomic_recipe.UnpackMode.NONE)
def get_attr(obj: object, name: str, /):
    """
    getattr doesn't have an inspectable signature until python 3.13, so provide a
    well-formed wrapper.
    """
    return getattr(obj, name)


@atomic_parser.atomic
def identity(x):
    return x
