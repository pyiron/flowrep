import unittest

from flowrep import base_models, edge_models, std
from flowrep.drawing import style
from flowrep.prospective import constant_recipe, for_recipe, helper_models


class TestPalette(unittest.TestCase):
    def test_every_recipe_type_has_colours(self):
        """A new RecipeElementType must not silently render uncoloured."""
        for kind in base_models.RecipeElementType:
            with self.subTest(kind=kind):
                self.assertIn(kind, style.NODE_PALETTE)

    def test_colours_are_hex_pairs(self):
        for kind, (fill, line) in style.NODE_PALETTE.items():
            with self.subTest(kind=kind):
                self.assertRegex(fill, r"^#[0-9a-f]{6}$")
                self.assertRegex(line, r"^#[0-9a-f]{6}$")

    def test_io_colour_is_not_a_node_colour(self):
        """IO grey must never be mistaken for a node type."""
        self.assertNotIn(
            style.IO_FILL, [fill for fill, _ in style.NODE_PALETTE.values()]
        )


class TestTruncation(unittest.TestCase):
    def test_truncate_right_leaves_short_text(self):
        self.assertEqual(style.truncate_right("abc", 10), "abc")

    def test_truncate_right_at_limit(self):
        self.assertEqual(style.truncate_right("abcde", 5), "abcde")

    def test_truncate_right_trims_tail(self):
        self.assertEqual(style.truncate_right("abcdefgh", 5), "abcd…")

    def test_truncate_left_leaves_short_text(self):
        self.assertEqual(style.truncate_left("abc", 10), "abc")

    def test_truncate_left_keeps_informative_tail(self):
        self.assertEqual(style.truncate_left("a.b.c.deep_name", 10), "…deep_name")


class TestWrapLabel(unittest.TestCase):
    def test_short_label_is_one_line(self):
        self.assertEqual(style.wrap_label("short", 20), ["short"])

    def test_breaks_after_underscore(self):
        self.assertEqual(
            style.wrap_label("some_quite_long_identifier_here", 12),
            ["some_quite_", "long_", "identifier_", "here"],
        )

    def test_hard_breaks_when_no_underscore_fits(self):
        self.assertEqual(style.wrap_label("abcdefghij", 4), ["abcd", "efgh", "ij"])


class TestFormatAnnotation(unittest.TestCase):
    def test_none_yields_none(self):
        self.assertIsNone(style.format_annotation(None))

    def test_builtin_by_bare_name(self):
        self.assertEqual(style.format_annotation(int), "int")

    def test_generic_keeps_args(self):
        self.assertEqual(style.format_annotation(list[float]), "list[float]")

    def test_class_by_qualname(self):
        class Widget: ...

        self.assertEqual(style.format_annotation(Widget), "Widget")

    def test_long_annotation_truncated(self):
        self.assertEqual(
            style.format_annotation(dict[str, list[tuple[int, float]]]),
            style.truncate_right(
                "dict[str, list[tuple[int, float]]]", style.ANNOTATION_MAX
            ),
        )


class TestSubtitle(unittest.TestCase):
    def test_atomic_uses_left_truncated_qualified_name(self):
        self.assertTrue(style.subtitle_for(std.neg.flowrep_recipe).endswith("neg"))

    def test_constant_uses_repr(self):
        self.assertEqual(
            style.subtitle_for(constant_recipe.ConstantRecipe(constant=42)), "42"
        )

    def test_no_reference_yields_none(self):
        """Flow-control recipes have no reference at all."""
        recipe = for_recipe.ForEachRecipe(
            inputs=["xs"],
            outputs=["ys"],
            body_node=helper_models.LabeledRecipe(
                label="body", recipe=std.neg.flowrep_recipe
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="a"
                ): edge_models.InputSource(port="xs")
            },
            output_edges={
                edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                    node="body", port="negative"
                )
            },
            nested_ports=["a"],
        )
        self.assertIsNone(style.subtitle_for(recipe))
