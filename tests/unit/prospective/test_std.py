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
        out = wfms.run_recipe(std.abs.node, a=-3)
        self.assertEqual(3, out.output_ports["absolute"].value)

    def test_add(self):
        out = wfms.run_recipe(std.add.node, a=1, b=2)
        self.assertEqual(3, out.output_ports["added"].value)

    def test_index(self):
        out = wfms.run_recipe(std.index.node, a=5)
        self.assertEqual(5, out.output_ports["index"].value)

    def test_inv(self):
        out = wfms.run_recipe(std.inv.node, a=0)
        self.assertEqual(-1, out.output_ports["inverted"].value)

    def test_invert(self):
        out = wfms.run_recipe(std.invert.node, a=0)
        self.assertEqual(-1, out.output_ports["inverted"].value)

    def test_neg(self):
        out = wfms.run_recipe(std.neg.node, a=2)
        self.assertEqual(-2, out.output_ports["negative"].value)

    def test_pos(self):
        out = wfms.run_recipe(std.pos.node, a=-2)
        self.assertEqual(-2, out.output_ports["positive"].value)

    def test_not_(self):
        out = wfms.run_recipe(std.not_.node, a=False)
        self.assertEqual(True, out.output_ports["negated"].value)

    def test_truth(self):
        out = wfms.run_recipe(std.truth.node, a=[])
        self.assertEqual(False, out.output_ports["truth"].value)

    def test_length_hint(self):
        out = wfms.run_recipe(std.length_hint.node, obj=[1, 2, 3])
        self.assertEqual(3, out.output_ports["length"].value)

    def test_sub(self):
        out = wfms.run_recipe(std.sub.node, a=5, b=2)
        self.assertEqual(3, out.output_ports["difference"].value)

    def test_isub(self):
        out = wfms.run_recipe(std.isub.node, a=5, b=2)
        self.assertEqual(3, out.output_ports["difference"].value)

    def test_iadd(self):
        out = wfms.run_recipe(std.iadd.node, a=1, b=2)
        self.assertEqual(3, out.output_ports["added"].value)

    def test_mul(self):
        out = wfms.run_recipe(std.mul.node, a=2, b=3)
        self.assertEqual(6, out.output_ports["product"].value)

    def test_imul(self):
        out = wfms.run_recipe(std.imul.node, a=2, b=3)
        self.assertEqual(6, out.output_ports["product"].value)

    def test_floordiv(self):
        out = wfms.run_recipe(std.floordiv.node, a=7, b=2)
        self.assertEqual(3, out.output_ports["quotient"].value)

    def test_ifloordiv(self):
        out = wfms.run_recipe(std.ifloordiv.node, a=7, b=2)
        self.assertEqual(3, out.output_ports["quotient"].value)

    def test_truediv(self):
        out = wfms.run_recipe(std.truediv.node, a=6, b=2)
        self.assertEqual(3.0, out.output_ports["quotient"].value)

    def test_itruediv(self):
        out = wfms.run_recipe(std.itruediv.node, a=6, b=2)
        self.assertEqual(3.0, out.output_ports["quotient"].value)

    def test_mod(self):
        out = wfms.run_recipe(std.mod.node, a=7, b=3)
        self.assertEqual(1, out.output_ports["remainder"].value)

    def test_imod(self):
        out = wfms.run_recipe(std.imod.node, a=7, b=3)
        self.assertEqual(1, out.output_ports["remainder"].value)

    def test_pow(self):
        out = wfms.run_recipe(std.pow.node, a=2, b=3)
        self.assertEqual(8, out.output_ports["power"].value)

    def test_ipow(self):
        out = wfms.run_recipe(std.ipow.node, a=2, b=3)
        self.assertEqual(8, out.output_ports["power"].value)

    def test_and_(self):
        out = wfms.run_recipe(std.and_.node, a=6, b=3)
        self.assertEqual(2, out.output_ports["conjunction"].value)

    def test_iand(self):
        out = wfms.run_recipe(std.iand.node, a=6, b=3)
        self.assertEqual(2, out.output_ports["conjunction"].value)

    def test_or_(self):
        out = wfms.run_recipe(std.or_.node, a=6, b=1)
        self.assertEqual(7, out.output_ports["disjunction"].value)

    def test_ior(self):
        out = wfms.run_recipe(std.ior.node, a=6, b=1)
        self.assertEqual(7, out.output_ports["disjunction"].value)

    def test_xor(self):
        out = wfms.run_recipe(std.xor.node, a=6, b=3)
        self.assertEqual(5, out.output_ports["exclusive_or"].value)

    def test_ixor(self):
        out = wfms.run_recipe(std.ixor.node, a=6, b=3)
        self.assertEqual(5, out.output_ports["exclusive_or"].value)

    def test_lshift(self):
        out = wfms.run_recipe(std.lshift.node, a=1, b=3)
        self.assertEqual(8, out.output_ports["left_shifted"].value)

    def test_ilshift(self):
        out = wfms.run_recipe(std.ilshift.node, a=1, b=3)
        self.assertEqual(8, out.output_ports["left_shifted"].value)

    def test_rshift(self):
        out = wfms.run_recipe(std.rshift.node, a=8, b=2)
        self.assertEqual(2, out.output_ports["right_shifted"].value)

    def test_irshift(self):
        out = wfms.run_recipe(std.irshift.node, a=8, b=2)
        self.assertEqual(2, out.output_ports["right_shifted"].value)

    def test_matmul(self):
        out = wfms.run_recipe(std.matmul.node, a=_Mat(1), b=_Mat(2))
        result = out.output_ports["matrix_product"].value
        self.assertEqual(("mm", 1, 2), result)

    def test_imatmul(self):
        out = wfms.run_recipe(std.imatmul.node, a=_Mat(1), b=_Mat(2))
        result = out.output_ports["matrix_product"].value
        self.assertEqual(("imm", 1, 2), result)

    def test_eq(self):
        out = wfms.run_recipe(std.eq.node, a=1, b=1)
        self.assertEqual(True, out.output_ports["equal"].value)

    def test_ne(self):
        out = wfms.run_recipe(std.ne.node, a=1, b=2)
        self.assertEqual(True, out.output_ports["not_equal"].value)

    def test_lt(self):
        out = wfms.run_recipe(std.lt.node, a=1, b=2)
        self.assertEqual(True, out.output_ports["less"].value)

    def test_le(self):
        out = wfms.run_recipe(std.le.node, a=2, b=2)
        self.assertEqual(True, out.output_ports["less_equal"].value)

    def test_gt(self):
        out = wfms.run_recipe(std.gt.node, a=2, b=1)
        self.assertEqual(True, out.output_ports["greater"].value)

    def test_ge(self):
        out = wfms.run_recipe(std.ge.node, a=2, b=2)
        self.assertEqual(True, out.output_ports["greater_equal"].value)

    def test_is_(self):
        out = wfms.run_recipe(std.is_.node, a=None, b=None)
        self.assertEqual(True, out.output_ports["identical"].value)

    def test_is_not(self):
        out = wfms.run_recipe(std.is_not.node, a=1, b=2)
        self.assertEqual(True, out.output_ports["not_identical"].value)

    def test_contains(self):
        out = wfms.run_recipe(std.contains.node, a=[1, 2, 3], b=2)
        self.assertEqual(True, out.output_ports["contains"].value)

    def test_countOf(self):
        out = wfms.run_recipe(std.countOf.node, a=[1, 2, 2, 3], b=2)
        self.assertEqual(2, out.output_ports["count"].value)

    def test_indexOf(self):
        out = wfms.run_recipe(std.indexOf.node, a=[1, 2, 3], b=3)
        self.assertEqual(2, out.output_ports["index"].value)

    def test_concat(self):
        out = wfms.run_recipe(std.concat.node, a=[1], b=[2])
        self.assertEqual([1, 2], out.output_ports["concatenated"].value)

    def test_iconcat(self):
        out = wfms.run_recipe(std.iconcat.node, a=[1], b=[2])
        self.assertEqual([1, 2], out.output_ports["concatenated"].value)

    def test_getitem(self):
        out = wfms.run_recipe(std.getitem.node, a=[10, 20], b=1)
        self.assertEqual(20, out.output_ports["item"].value)

    def test_setitem(self):
        d = {}
        wfms.run_recipe(std.setitem.node, a=d, b="k", c=1)
        self.assertEqual(1, d["k"])

    def test_delitem(self):
        d = {"k": 1}
        wfms.run_recipe(std.delitem.node, a=d, b="k")
        self.assertNotIn("k", d)

    def test_call(self):
        with self.subTest("no variadics"):
            out = wfms.run_recipe(std.call.node, obj=_return_42)
            self.assertEqual(42, out.output_ports["result"].value)
        with self.subTest("args"):
            out = wfms.run_recipe(std.call.node, obj=_return_42, args_=(1, 2))
            self.assertEqual((42, (1, 2)), out.output_ports["result"].value)
        with self.subTest("kwargs"):
            out = wfms.run_recipe(
                std.call.node, obj=_return_42, kwargs_={"a": 3, "b": 4}
            )
            self.assertEqual((42, {"a": 3, "b": 4}), out.output_ports["result"].value)
        with self.subTest("both"):
            out = wfms.run_recipe(
                std.call.node, obj=_return_42, args_=(1, 2), kwargs_={"a": 3, "b": 4}
            )
            self.assertEqual(
                (42, (1, 2), {"a": 3, "b": 4}), out.output_ports["result"].value
            )

    def test_getattr_(self):
        with self.subTest("instance attribute"):
            out = wfms.run_recipe(
                std.getattr_.node, obj=_HasAttrs(), name="instance_attr"
            )
            self.assertEqual(42, out.output_ports["attr"].value)

        with self.subTest("class attribute"):
            out = wfms.run_recipe(std.getattr_.node, obj=_HasAttrs(), name="class_attr")
            self.assertEqual("shared", out.output_ports["attr"].value)

        with self.subTest("bound method"):
            out = wfms.run_recipe(std.getattr_.node, obj=_HasAttrs(), name="method")
            self.assertEqual("called", out.output_ports["attr"].value())

        with self.subTest("tuple attributes are delivered whole"):
            out = wfms.run_recipe(std.getattr_.node, obj=_HasAttrs(), name="tuple_attr")
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
            wfms.run_recipe(std.getattr_.node, obj=_HasAttrs(), name="not_here")

    def test_identity(self):
        with self.subTest("value"):
            out = wfms.run_recipe(std.identity.node, x=42)
            self.assertEqual(42, out.output_ports["x"].value)

        with self.subTest("none"):
            out = wfms.run_recipe(std.identity.node, x=None)
            self.assertIsNone(out.output_ports["x"].value)

        with self.subTest("tuples are delivered whole"):
            out = wfms.run_recipe(std.identity.node, x=(1, 2))
            self.assertEqual(
                (1, 2),
                out.output_ports["x"].value,
                msg="A tuple input should land in the single output port intact, "
                "rather than being unpacked across ports",
            )

        with self.subTest("passthrough is not a copy"):
            obj = _HasAttrs()
            out = wfms.run_recipe(std.identity.node, x=obj)
            self.assertIs(
                obj,
                out.output_ports["x"].value,
                msg="Identity should pass the very same object through",
            )


if __name__ == "__main__":
    unittest.main()
