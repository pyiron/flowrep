"""Helper functions parsed in multiple test files"""

from flowrep.models.parsers import atomic_parser


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
