import enum
import typing
import unittest

from flowrep import _annotation_source

from flowrep_static import library


class TestRenderDefault(unittest.TestCase):
    def test_inlines_simple_literals(self):
        cases = [0, -3, 3.5, "hi", b"by", True, False, None]
        for value in cases:
            with self.subTest(value=value):
                text = _annotation_source.render_default(value)
                self.assertIsNotNone(text)
                self.assertEqual(eval(text), value)  # noqa: S307 - controlled text

    def test_inlines_nested_containers(self):
        value = {"a": [1, 2], "b": (3, {"c": 4}), "d": {5, 6}}
        text = _annotation_source.render_default(value)
        self.assertIsNotNone(text)
        self.assertEqual(eval(text), value)  # noqa: S307 - controlled text

    def test_frozenset_falls_back(self):
        # frozenset has no literal syntax (repr is a call), so ast.literal_eval
        # rejects it and it stays namespace-bound. This is intentional.
        self.assertIsNone(_annotation_source.render_default(frozenset({1, 2})))

    def test_rejects_object_sentinel(self):
        self.assertIsNone(_annotation_source.render_default(object()))

    def test_rejects_nan(self):
        self.assertIsNone(_annotation_source.render_default(float("nan")))

    def test_rejects_enum_member(self):
        class Color(enum.Enum):
            RED = 1

        self.assertIsNone(_annotation_source.render_default(Color.RED))

    def test_rejects_arbitrary_instance(self):
        class Weird:
            pass

        self.assertIsNone(_annotation_source.render_default(Weird()))

    def test_rejects_instance_with_literal_looking_repr(self):
        # repr parses as a literal (42) but the value is not that type, so the
        # type/equality guard rejects it (the literal_eval-succeeds branch).
        class FakeInt:
            def __repr__(self):
                return "42"

        self.assertIsNone(_annotation_source.render_default(FakeInt()))


class TestRenderAnnotationPlain(unittest.TestCase):
    def test_builtin_needs_no_import(self):
        imports: set[str] = set()
        self.assertEqual(_annotation_source.render_annotation(int, imports), "int")
        self.assertEqual(imports, set())

    def test_none_renders_as_none(self):
        imports: set[str] = set()
        self.assertEqual(_annotation_source.render_annotation(None, imports), "None")
        self.assertEqual(
            _annotation_source.render_annotation(type(None), imports), "None"
        )

    def test_non_builtin_class_records_import(self):
        import decimal

        imports: set[str] = set()
        text = _annotation_source.render_annotation(decimal.Decimal, imports)
        self.assertEqual(text, "decimal.Decimal")
        self.assertEqual(imports, {"import decimal"})

    def test_class_renders_with_import(self):
        imports: set[str] = set()
        text = _annotation_source.render_annotation(library.MyCustomException, imports)
        self.assertEqual(text, "flowrep_static.library.MyCustomException")
        self.assertEqual(imports, {"import flowrep_static.library"})

    def test_local_class_falls_back(self):
        class Local:
            pass

        imports: set[str] = set()
        self.assertIsNone(_annotation_source.render_annotation(Local, imports))

    def test_arbitrary_instance_falls_back(self):
        imports: set[str] = set()
        self.assertIsNone(_annotation_source.render_annotation(object(), imports))


class TestRenderAnnotationCompound(unittest.TestCase):
    def _ok(self, ann, expected_eval):
        imports: set[str] = set()
        text = _annotation_source.render_annotation(ann, imports)
        self.assertIsNotNone(text, msg=f"{ann!r} should render")
        ns = {"typing": typing}
        for line in imports:
            mod = line.removeprefix("import ").split(".")[0]
            ns[mod] = __import__(mod)
        self.assertEqual(eval(text, ns), expected_eval)  # noqa: S307
        return text

    def test_builtin_generic(self):
        self._ok(list[int], list[int])

    def test_dict_generic(self):
        self._ok(dict[str, int], dict[str, int])

    def test_tuple_generic(self):
        self._ok(tuple[int, float], tuple[int, float])

    def test_union(self):
        text = self._ok(int | str, int | str)
        self.assertIn("|", text)

    def test_optional(self):
        # typing.Optional[int] is int | None; both compare equal.
        self._ok(typing.Optional[int], typing.Optional[int])  # noqa: UP045

    def test_annotated_literal_metadata(self):
        self._ok(typing.Annotated[int, "meta"], typing.Annotated[int, "meta"])

    def test_literal(self):
        self._ok(typing.Literal[1, "x", True], typing.Literal[1, "x", True])

    def test_nested_generic(self):
        self._ok(dict[str, list[int]], dict[str, list[int]])

    def test_annotated_nonliteral_metadata_falls_back(self):
        imports: set[str] = set()
        meta = object()
        self.assertIsNone(
            _annotation_source.render_annotation(typing.Annotated[int, meta], imports)
        )

    def test_unrenderable_arg_falls_back(self):
        class Local:
            pass

        imports: set[str] = set()
        self.assertIsNone(_annotation_source.render_annotation(list[Local], imports))


class TestRenderAnnotationFallbacks(unittest.TestCase):
    def test_union_with_unrenderable_member_falls_back(self):
        class Local:
            pass

        imports: set[str] = set()
        self.assertIsNone(_annotation_source.render_annotation(int | Local, imports))

    def test_annotated_unrenderable_base_falls_back(self):
        class Local:
            pass

        imports: set[str] = set()
        self.assertIsNone(
            _annotation_source.render_annotation(
                typing.Annotated[Local, "meta"], imports
            )
        )

    def test_literal_with_enum_member_falls_back(self):
        class Color(enum.Enum):
            RED = 1

        imports: set[str] = set()
        self.assertIsNone(
            _annotation_source.render_annotation(typing.Literal[Color.RED], imports)
        )

    def test_local_generic_origin_falls_back(self):
        T = typing.TypeVar("T")

        class G(typing.Generic[T]):
            pass

        imports: set[str] = set()
        self.assertIsNone(_annotation_source.render_annotation(G[int], imports))

    def test_unsupported_special_form_falls_back(self):
        # typing.ClassVar[int] has a non-None origin that is not a type and not
        # one of the handled special forms, so it falls back.
        imports: set[str] = set()
        self.assertIsNone(
            _annotation_source.render_annotation(typing.ClassVar[int], imports)
        )


if __name__ == "__main__":
    unittest.main()
