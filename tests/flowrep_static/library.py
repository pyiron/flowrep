"""Helper functions parsed in multiple test files"""

import typing

from flowrep.parsers import atomic_parser, dataclass_parser, workflow_parser


def undecorated_identity(x):
    """For checking parser propagation"""
    return x


def multi_result(x):
    """Undecorated; parsed on-the-fly with two outputs."""
    a = x + 1
    b = x - 1
    return a, b


@atomic_parser.atomic
def identity(x):
    return x


@atomic_parser.atomic
def my_range(n):
    return list(range(n))


@atomic_parser.atomic
def my_condition(m, n):
    return m < n


@atomic_parser.atomic
def my_add(a, b):
    return a + b


@atomic_parser.atomic
def my_mul(a, b):
    return a * b


# test_workflow_parser


def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


def multiply(x: float, y: float = 5) -> float:
    return x * y


@atomic_parser.atomic
def negate(x):
    return -x


@atomic_parser.atomic
def increment(x, step=1):
    return x + step


@atomic_parser.atomic
def decrement(x: int) -> int:
    return x - 1


@atomic_parser.atomic
def is_positive(n):
    return n > 0


@atomic_parser.atomic
def divide(a, b):
    return a / b


@atomic_parser.atomic
def divmod_func(a: float, b: float) -> tuple[float, float]:
    quotient = a // b
    remainder = a % b
    return quotient, remainder


@workflow_parser.workflow
def simple_workflow(a, b):
    result = add(a, b)
    return result


def no_input_atomic():
    return 42


@workflow_parser.workflow
def no_input_workflow():
    a = no_input_atomic()
    return a


class MyCustomException(ValueError): ...


@atomic_parser.atomic
def raises_custom(x, y):
    raise MyCustomException("Custom exception")
    return x + y


@atomic_parser.atomic
def labeled_x(seed) -> typing.Annotated[int, {"label": "x"}]:
    x = seed + 1
    return x


@atomic_parser.atomic
def loop_inc(x):
    y = x + 1
    return y


@atomic_parser.atomic
def combine(a, b):
    r = a + b
    return r


@atomic_parser.atomic
def split_pair(
    v,
) -> tuple[
    typing.Annotated[int, {"label": "lo"}],
    typing.Annotated[int, {"label": "hi"}],
]:
    lo = v
    hi = v + 1
    return lo, hi


@atomic_parser.atomic
def make_list(seed) -> typing.Annotated[list, {"label": "data"}]:
    data = [seed, seed + 1]
    return data


@workflow_parser.workflow
def macro_identity(x):
    return x


# test_attribute_parser / test_workflow_parser / test_compiler / integration


class ComplexData:
    """A plain (non-recipe) payload class, reached via attribute access."""

    def __init__(self, val: int = 0):
        self.val = val


@dataclass_parser.dataclass
class MyDataclass:
    a: ComplexData
    x: int = 1
