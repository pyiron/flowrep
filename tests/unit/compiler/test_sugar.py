import unittest

from flowrep import edge_models, std
from flowrep.compiler import sugar
from flowrep.parsers import workflow_parser
from flowrep.prospective import constant_recipe

from flowrep_static import library, makers


def _wf(x0: int, comp: library.ComplexData):
    dc = library.MyDataclass(comp, x0)
    v = dc.a
    return v


class TestIsStdGetattr(unittest.TestCase):
    def test_true_for_std_getattr_recipe(self):
        self.assertTrue(sugar.is_std_getattr(std.get_attr.flowrep_recipe))

    def test_false_for_other_atomic(self):
        self.assertFalse(sugar.is_std_getattr(std.add.flowrep_recipe))

    def test_false_for_constant_recipe(self):
        self.assertFalse(
            sugar.is_std_getattr(constant_recipe.ConstantRecipe(constant=1))
        )

    def test_false_for_workflow_recipe(self):
        recipe = makers.reference_free(_wf)
        self.assertFalse(sugar.is_std_getattr(recipe))


class TestIsAttributeSyntax(unittest.TestCase):
    def test_true_cases(self):
        for value in ("a", "inputs", "_x", "__class__"):
            with self.subTest(value):
                self.assertTrue(sugar.is_attribute_syntax(value))

    def test_false_for_keyword(self):
        self.assertFalse(sugar.is_attribute_syntax("class"))

    def test_false_for_non_identifiers(self):
        for value in ("1a", "a b", ""):
            with self.subTest(value):
                self.assertFalse(sugar.is_attribute_syntax(value))

    def test_false_for_non_string_types(self):
        for value in (3, None):
            with self.subTest(value):
                self.assertFalse(sugar.is_attribute_syntax(value))


class TestAttributeName(unittest.TestCase):
    def test_returns_name_for_parsed_access(self):
        recipe = makers.reference_free(_wf)
        self.assertEqual(sugar.attribute_name("getattr_a_0", recipe), "a")

    def test_none_when_name_port_fed_by_non_constant_node(self):
        recipe = workflow_parser.parse_workflow(_wf).model_copy(
            update={"reference": None}
        )
        # Rewire the name port to come from a non-constant node output instead of a
        # constant peer -- a shape only hand-construction produces.
        target = edge_models.TargetHandle(node="getattr_a_0", port="name")
        new_edges = dict(recipe.edges)
        new_edges[target] = edge_models.SourceHandle(
            node="MyDataclass_0", port="instance"
        )
        recipe = recipe.model_copy(update={"edges": new_edges})
        self.assertIsNone(sugar.attribute_name("getattr_a_0", recipe))

    def test_none_when_constant_is_non_identifier_string(self):
        recipe = workflow_parser.parse_workflow(_wf).model_copy(
            update={"reference": None}
        )
        new_nodes = dict(recipe.nodes)
        new_nodes["constant_0"] = constant_recipe.ConstantRecipe(constant="not an id!")
        recipe = recipe.model_copy(update={"nodes": new_nodes})
        self.assertIsNone(sugar.attribute_name("getattr_a_0", recipe))

    def test_none_when_name_port_unwired(self):
        recipe = workflow_parser.parse_workflow(_wf).model_copy(
            update={"reference": None}
        )
        target = edge_models.TargetHandle(node="getattr_a_0", port="name")
        new_edges = {k: v for k, v in recipe.edges.items() if k != target}
        recipe = recipe.model_copy(update={"edges": new_edges})
        self.assertIsNone(sugar.attribute_name("getattr_a_0", recipe))


class TestPortConstantsTrackTheRecipe(unittest.TestCase):
    """Editing std.getattr_'s ports must not require editing strings in src/."""

    def test_input_ports_come_from_the_recipe(self):
        self.assertEqual(
            (sugar.OBJ_PORT, sugar.NAME_PORT), tuple(std.get_attr.flowrep_recipe.inputs)
        )

    def test_output_port_comes_from_the_recipe(self):
        self.assertEqual((sugar.ATTR_PORT,), tuple(std.get_attr.flowrep_recipe.outputs))


if __name__ == "__main__":
    unittest.main()
