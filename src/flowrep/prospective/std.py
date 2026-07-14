"""
A standard library of flowrep labeled recipes built around fundamental python
operations and most common use cases.
"""

import operator
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from pyiron_snippets import versions

from flowrep import base_models
from flowrep.prospective import atomic_recipe, helper_models

abs = helper_models.LabeledRecipe(
    label="abs",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.abs),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a"],
        outputs=["absolute"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

add = helper_models.LabeledRecipe(
    label="add",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.add),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["added"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

index = helper_models.LabeledRecipe(
    label="index",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.index),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a"],
        outputs=["index"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

inv = helper_models.LabeledRecipe(
    label="inv",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.inv),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a"],
        outputs=["inverted"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

invert = helper_models.LabeledRecipe(
    label="invert",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.invert),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a"],
        outputs=["inverted"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

neg = helper_models.LabeledRecipe(
    label="neg",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.neg),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a"],
        outputs=["negative"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

pos = helper_models.LabeledRecipe(
    label="pos",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.pos),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a"],
        outputs=["positive"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

not_ = helper_models.LabeledRecipe(
    label="not_",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.not_),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a"],
        outputs=["negated"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

truth = helper_models.LabeledRecipe(
    label="truth",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.truth),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a"],
        outputs=["truth"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

length_hint = helper_models.LabeledRecipe(
    label="length_hint",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.length_hint),
            restricted_input_kinds={
                "obj": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["obj"],
        outputs=["length"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

sub = helper_models.LabeledRecipe(
    label="sub",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.sub),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["difference"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

isub = helper_models.LabeledRecipe(
    label="isub",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.isub),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["difference"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

iadd = helper_models.LabeledRecipe(
    label="iadd",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.iadd),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["added"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

mul = helper_models.LabeledRecipe(
    label="mul",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.mul),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["product"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

imul = helper_models.LabeledRecipe(
    label="imul",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.imul),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["product"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

floordiv = helper_models.LabeledRecipe(
    label="floordiv",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.floordiv),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["quotient"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

ifloordiv = helper_models.LabeledRecipe(
    label="ifloordiv",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.ifloordiv),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["quotient"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

truediv = helper_models.LabeledRecipe(
    label="truediv",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.truediv),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["quotient"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

itruediv = helper_models.LabeledRecipe(
    label="itruediv",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.itruediv),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["quotient"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

mod = helper_models.LabeledRecipe(
    label="mod",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.mod),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["remainder"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

imod = helper_models.LabeledRecipe(
    label="imod",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.imod),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["remainder"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

pow = helper_models.LabeledRecipe(
    label="pow",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.pow),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["power"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

ipow = helper_models.LabeledRecipe(
    label="ipow",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.ipow),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["power"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

and_ = helper_models.LabeledRecipe(
    label="and_",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.and_),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["conjunction"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

iand = helper_models.LabeledRecipe(
    label="iand",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.iand),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["conjunction"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

or_ = helper_models.LabeledRecipe(
    label="or_",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.or_),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["disjunction"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

ior = helper_models.LabeledRecipe(
    label="ior",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.ior),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["disjunction"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

xor = helper_models.LabeledRecipe(
    label="xor",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.xor),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["exclusive_or"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

ixor = helper_models.LabeledRecipe(
    label="ixor",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.ixor),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["exclusive_or"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

lshift = helper_models.LabeledRecipe(
    label="lshift",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.lshift),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["left_shifted"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

ilshift = helper_models.LabeledRecipe(
    label="ilshift",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.ilshift),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["left_shifted"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

rshift = helper_models.LabeledRecipe(
    label="rshift",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.rshift),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["right_shifted"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

irshift = helper_models.LabeledRecipe(
    label="irshift",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.irshift),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["right_shifted"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

matmul = helper_models.LabeledRecipe(
    label="matmul",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.matmul),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["matrix_product"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

imatmul = helper_models.LabeledRecipe(
    label="imatmul",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.imatmul),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["matrix_product"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

eq = helper_models.LabeledRecipe(
    label="eq",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.eq),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["equal"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

ne = helper_models.LabeledRecipe(
    label="ne",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.ne),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["not_equal"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

lt = helper_models.LabeledRecipe(
    label="lt",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.lt),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["less"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

le = helper_models.LabeledRecipe(
    label="le",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.le),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["less_equal"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

gt = helper_models.LabeledRecipe(
    label="gt",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.gt),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["greater"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

ge = helper_models.LabeledRecipe(
    label="ge",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.ge),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["greater_equal"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

is_ = helper_models.LabeledRecipe(
    label="is_",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.is_),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["identical"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

is_not = helper_models.LabeledRecipe(
    label="is_not",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.is_not),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["not_identical"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

contains = helper_models.LabeledRecipe(
    label="contains",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.contains),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["contains"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

countOf = helper_models.LabeledRecipe(
    label="countOf",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.countOf),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["count"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

indexOf = helper_models.LabeledRecipe(
    label="indexOf",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.indexOf),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["index"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

concat = helper_models.LabeledRecipe(
    label="concat",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.concat),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["concatenated"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

iconcat = helper_models.LabeledRecipe(
    label="iconcat",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.iconcat),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["concatenated"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

getitem = helper_models.LabeledRecipe(
    label="getitem",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.getitem),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["item"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

setitem = helper_models.LabeledRecipe(
    label="setitem",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.setitem),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "c": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b", "c"],
        outputs=["set"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)

delitem = helper_models.LabeledRecipe(
    label="delitem",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(operator.delitem),
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["a", "b"],
        outputs=["deleted"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)


def _call_wrapper(
    obj: Callable,
    args_: Iterable[Any] | None = None,
    kwargs_: Mapping[str, Any] | None = None,
):
    args_ = () if args_ is None else args_
    kwargs_ = {} if kwargs_ is None else kwargs_
    return obj(*args_, **kwargs_)


call = helper_models.LabeledRecipe(
    label="call",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(_call_wrapper),
            restricted_input_kinds={
                "obj": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "args_": base_models.RestrictedParamKind.KEYWORD_ONLY,
                "kwargs_": base_models.RestrictedParamKind.KEYWORD_ONLY,
            },
        ),
        inputs=["obj", "args_", "kwargs_"],
        outputs=["result"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)


def _getattr_wrapper(
    obj: object,
    name: str,
):
    """
    getattr doesn't have an inspectable signature until python 3.13, so provide a
    well-formed wrapper.
    """
    return getattr(obj, name)


getattr_ = helper_models.LabeledRecipe(
    label="getattr_",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(_getattr_wrapper),
            restricted_input_kinds={
                "obj": base_models.RestrictedParamKind.POSITIONAL_ONLY,
                "name": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        ),
        inputs=["obj", "name"],
        outputs=["attr"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)


def _identity_function(x):
    return x


identity = helper_models.LabeledRecipe(
    label="identity",
    recipe=atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=versions.VersionInfo.of(_identity_function),
        ),
        inputs=["x"],
        outputs=["x"],
        unpack_mode=atomic_recipe.UnpackMode.NONE,
    ),
)
