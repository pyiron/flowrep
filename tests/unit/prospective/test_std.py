from __future__ import annotations

import unittest

from flowrep import wfms
from flowrep.prospective import std


def _return_42(*args, **kwargs):
    if args:
        if kwargs:
            return 42, args, kwargs
        return 42, args
    elif kwargs:
        return 42, kwargs
    return 42


class _Mat:
    def __init__(self, v):
        self.v = v

    def __matmul__(self, other):
        return ("mm", self.v, other.v)

    def __imatmul__(self, other):
        return ("imm", self.v, other.v)


class _HasAttrs:
    class_attr = "shared"

    def __init__(self):
        self.instance_attr = 42
        self.tuple_attr = (1, 2)

    def method(self):
        return "called"


class TestStdExecution(unittest.TestCase):

    def test_abs(self):
        out = wfms.run_recipe(std.abs.flowrep_recipe, a=-3)
        self.assertEqual(3, out.output_ports["absolute"].value)

    def test_add(self):
        out = wfms.run_recipe(std.add.recipe, a=1, b=2)
        self.assertEqual(3, out.output_ports["added"].value)

    def test_index(self):
        out = wfms.run_recipe(std.index.recipe, a=5)
        self.assertEqual(5, out.output_ports["index"].value)

    def test_inv(self):
        out = wfms.run_recipe(std.inv.recipe, a=0)
        self.assertEqual(-1, out.output_ports["inverted"].value)

    def test_invert(self):
        out = wfms.run_recipe(std.invert.recipe, a=0)
        self.assertEqual(-1, out.output_ports["inverted"].value)

    def test_neg(self):
        out = wfms.run_recipe(std.neg.recipe, a=2)
        self.assertEqual(-2, out.output_ports["negative"].value)

    def test_pos(self):
        out = wfms.run_recipe(std.pos.recipe, a=-2)
        self.assertEqual(-2, out.output_ports["positive"].value)

    def test_not_(self):
        out = wfms.run_recipe(std.not_.recipe, a=False)
        self.assertEqual(True, out.output_ports["negated"].value)

    def test_truth(self):
        out = wfms.run_recipe(std.truth.recipe, a=[])
        self.assertEqual(False, out.output_ports["truth"].value)

    def test_length_hint(self):
        out = wfms.run_recipe(std.length_hint.recipe, obj=[1, 2, 3])
        self.assertEqual(3, out.output_ports["length"].value)

    def test_sub(self):
        out = wfms.run_recipe(std.sub.recipe, a=5, b=2)
        self.assertEqual(3, out.output_ports["difference"].value)

    def test_isub(self):
        out = wfms.run_recipe(std.isub.recipe, a=5, b=2)
        self.assertEqual(3, out.output_ports["difference"].value)

    def test_iadd(self):
        out = wfms.run_recipe(std.iadd.recipe, a=1, b=2)
        self.assertEqual(3, out.output_ports["added"].value)

    def test_mul(self):
        out = wfms.run_recipe(std.mul.recipe, a=2, b=3)
        self.assertEqual(6, out.output_ports["product"].value)

    def test_imul(self):
        out = wfms.run_recipe(std.imul.recipe, a=2, b=3)
        self.assertEqual(6, out.output_ports["product"].value)

    def test_floordiv(self):
        out = wfms.run_recipe(std.floordiv.recipe, a=7, b=2)
        self.assertEqual(3, out.output_ports["quotient"].value)

    def test_ifloordiv(self):
        out = wfms.run_recipe(std.ifloordiv.recipe, a=7, b=2)
        self.assertEqual(3, out.output_ports["quotient"].value)

    def test_truediv(self):
        out = wfms.run_recipe(std.truediv.recipe, a=6, b=2)
        self.assertEqual(3.0, out.output_ports["quotient"].value)

    def test_itruediv(self):
        out = wfms.run_recipe(std.itruediv.recipe, a=6, b=2)
        self.assertEqual(3.0, out.output_ports["quotient"].value)

    def test_mod(self):
        out = wfms.run_recipe(std.mod.recipe, a=7, b=3)
        self.assertEqual(1, out.output_ports["remainder"].value)

    def test_imod(self):
        out = wfms.run_recipe(std.imod.recipe, a=7, b=3)
        self.assertEqual(1, out.output_ports["remainder"].value)

    def test_pow(self):
        out = wfms.run_recipe(std.pow.recipe, a=2, b=3)
        self.assertEqual(8, out.output_ports["power"].value)

    def test_ipow(self):
        out = wfms.run_recipe(std.ipow.recipe, a=2, b=3)
        self.assertEqual(8, out.output_ports["power"].value)

    def test_and_(self):
        out = wfms.run_recipe(std.and_.recipe, a=6, b=3)
        self.assertEqual(2, out.output_ports["conjunction"].value)

    def test_iand(self):
        out = wfms.run_recipe(std.iand.recipe, a=6, b=3)
        self.assertEqual(2, out.output_ports["conjunction"].value)

    def test_or_(self):
        out = wfms.run_recipe(std.or_.recipe, a=6, b=1)
        self.assertEqual(7, out.output_ports["disjunction"].value)

    def test_ior(self):
        out = wfms.run_recipe(std.ior.recipe, a=6, b=1)
        self.assertEqual(7, out.output_ports["disjunction"].value)

    def test_xor(self):
        out = wfms.run_recipe(std.xor.recipe, a=6, b=3)
        self.assertEqual(5, out.output_ports["exclusive_or"].value)

    def test_ixor(self):
        out = wfms.run_recipe(std.ixor.recipe, a=6, b=3)
        self.assertEqual(5, out.output_ports["exclusive_or"].value)

    def test_lshift(self):
        out = wfms.run_recipe(std.lshift.recipe, a=1, b=3)
        self.assertEqual(8, out.output_ports["left_shifted"].value)

    def test_ilshift(self):
        out = wfms.run_recipe(std.ilshift.recipe, a=1, b=3)
        self.assertEqual(8, out.output_ports["left_shifted"].value)

    def test_rshift(self):
        out = wfms.run_recipe(std.rshift.recipe, a=8, b=2)
        self.assertEqual(2, out.output_ports["right_shifted"].value)

    def test_irshift(self):
        out = wfms.run_recipe(std.irshift.recipe, a=8, b=2)
        self.assertEqual(2, out.output_ports["right_shifted"].value)

    def test_matmul(self):
        out = wfms.run_recipe(std.matmul.recipe, a=_Mat(1), b=_Mat(2))
        result = out.output_ports["matrix_product"].value
        self.assertEqual(("mm", 1, 2), result)

    def test_imatmul(self):
        out = wfms.run_recipe(std.imatmul.recipe, a=_Mat(1), b=_Mat(2))
        result = out.output_ports["matrix_product"].value
        self.assertEqual(("imm", 1, 2), result)

    def test_eq(self):
        out = wfms.run_recipe(std.eq.recipe, a=1, b=1)
        self.assertEqual(True, out.output_ports["equal"].value)

    def test_ne(self):
        out = wfms.run_recipe(std.ne.recipe, a=1, b=2)
        self.assertEqual(True, out.output_ports["not_equal"].value)

    def test_lt(self):
        out = wfms.run_recipe(std.lt.recipe, a=1, b=2)
        self.assertEqual(True, out.output_ports["less"].value)

    def test_le(self):
        out = wfms.run_recipe(std.le.recipe, a=2, b=2)
        self.assertEqual(True, out.output_ports["less_equal"].value)

    def test_gt(self):
        out = wfms.run_recipe(std.gt.recipe, a=2, b=1)
        self.assertEqual(True, out.output_ports["greater"].value)

    def test_ge(self):
        out = wfms.run_recipe(std.ge.recipe, a=2, b=2)
        self.assertEqual(True, out.output_ports["greater_equal"].value)

    def test_is_(self):
        out = wfms.run_recipe(std.is_.recipe, a=None, b=None)
        self.assertEqual(True, out.output_ports["identical"].value)

    def test_is_not(self):
        out = wfms.run_recipe(std.is_not.recipe, a=1, b=2)
        self.assertEqual(True, out.output_ports["not_identical"].value)

    def test_contains(self):
        out = wfms.run_recipe(std.contains.recipe, a=[1, 2, 3], b=2)
        self.assertEqual(True, out.output_ports["contains"].value)

    def test_countOf(self):
        out = wfms.run_recipe(std.countOf.recipe, a=[1, 2, 2, 3], b=2)
        self.assertEqual(2, out.output_ports["count"].value)

    def test_indexOf(self):
        out = wfms.run_recipe(std.indexOf.recipe, a=[1, 2, 3], b=3)
        self.assertEqual(2, out.output_ports["index"].value)

    def test_concat(self):
        out = wfms.run_recipe(std.concat.recipe, a=[1], b=[2])
        self.assertEqual([1, 2], out.output_ports["concatenated"].value)

    def test_iconcat(self):
        out = wfms.run_recipe(std.iconcat.recipe, a=[1], b=[2])
        self.assertEqual([1, 2], out.output_ports["concatenated"].value)

    def test_getitem(self):
        out = wfms.run_recipe(std.getitem.recipe, a=[10, 20], b=1)
        self.assertEqual(20, out.output_ports["item"].value)

    def test_setitem(self):
        d = {}
        wfms.run_recipe(std.setitem.recipe, a=d, b="k", c=1)
        self.assertEqual(1, d["k"])

    def test_delitem(self):
        d = {"k": 1}
        wfms.run_recipe(std.delitem.recipe, a=d, b="k")
        self.assertNotIn("k", d)

    def test_call(self):
        with self.subTest("no variadics"):
            out = wfms.run_recipe(std.call.flowrep_recipe, obj=_return_42)
            self.assertEqual(42, out.output_ports["result"].value)
        with self.subTest("args"):
            out = wfms.run_recipe(std.call.flowrep_recipe, obj=_return_42, args_=(1, 2))
            self.assertEqual((42, (1, 2)), out.output_ports["result"].value)
        with self.subTest("kwargs"):
            out = wfms.run_recipe(
                std.call.flowrep_recipe, obj=_return_42, kwargs_={"a": 3, "b": 4}
            )
            self.assertEqual((42, {"a": 3, "b": 4}), out.output_ports["result"].value)
        with self.subTest("both"):
            out = wfms.run_recipe(
                std.call.flowrep_recipe,
                obj=_return_42,
                args_=(1, 2),
                kwargs_={"a": 3, "b": 4},
            )
            self.assertEqual(
                (42, (1, 2), {"a": 3, "b": 4}), out.output_ports["result"].value
            )

    def test_getattr_(self):
        with self.subTest("instance attribute"):
            out = wfms.run_recipe(
                std.get_attr.flowrep_recipe, obj=_HasAttrs(), name="instance_attr"
            )
            self.assertEqual(42, out.output_ports["attr"].value)

        with self.subTest("class attribute"):
            out = wfms.run_recipe(
                std.get_attr.flowrep_recipe, obj=_HasAttrs(), name="class_attr"
            )
            self.assertEqual("shared", out.output_ports["attr"].value)

        with self.subTest("bound method"):
            out = wfms.run_recipe(
                std.get_attr.flowrep_recipe, obj=_HasAttrs(), name="method"
            )
            self.assertEqual("called", out.output_ports["attr"].value())

        with self.subTest("tuple attributes are delivered whole"):
            out = wfms.run_recipe(
                std.get_attr.flowrep_recipe, obj=_HasAttrs(), name="tuple_attr"
            )
            self.assertEqual(
                (1, 2),
                out.output_ports["attr"].value,
                msg="The single output port should hold the entire attribute, even "
                "when that attribute is itself a tuple",
            )

        with (
            self.subTest("missing attribute"),
            self.assertRaises(
                AttributeError, msg="Failed lookups should surface to the caller"
            ),
        ):
            wfms.run_recipe(
                std.get_attr.flowrep_recipe, obj=_HasAttrs(), name="not_here"
            )

    def test_identity(self):
        with self.subTest("value"):
            out = wfms.run_recipe(std.identity.flowrep_recipe, x=42)
            self.assertEqual(42, out.output_ports["x"].value)

        with self.subTest("none"):
            out = wfms.run_recipe(std.identity.flowrep_recipe, x=None)
            self.assertIsNone(out.output_ports["x"].value)

        with self.subTest("tuples are delivered whole"):
            out = wfms.run_recipe(std.identity.flowrep_recipe, x=(1, 2))
            self.assertEqual(
                (1, 2),
                out.output_ports["x"].value,
                msg="A tuple input should land in the single output port intact, "
                "rather than being unpacked across ports",
            )

        with self.subTest("passthrough is not a copy"):
            obj = _HasAttrs()
            out = wfms.run_recipe(std.identity.flowrep_recipe, x=obj)
            self.assertIs(
                obj,
                out.output_ports["x"].value,
                msg="Identity should pass the very same object through",
            )


if __name__ == "__main__":
    unittest.main()
