import unittest

from flowrep import edge_models, std
from flowrep.parsers import symbol_scope, workflow_parser
from flowrep.prospective import constant_recipe

from flowrep_static import library


def kinetic_energy(mass, velocity):
    v_2 = std.mul(velocity, velocity)
    mv_2 = std.mul(mass, v_2)
    ke = std.mul(0.5, mv_2)
    return ke


def keyword_constant(x):
    y = library.increment(x, step=3)
    return y


def compound_constant(x):
    y = std.mul(x, [1.0, 1, "1", {"key": [42]}])
    return y


def tuple_arg(x):
    y = std.mul(x, (1, 2))
    return y


def lambda_arg(x):
    y = std.mul(x, lambda z: z)
    return y


class TestConstantParsing(unittest.TestCase):
    def test_positional_literal_injects_constant(self):
        recipe = workflow_parser.parse_workflow(kinetic_energy)
        self.assertIn("constant_0", recipe.nodes)
        const = recipe.nodes["constant_0"]
        self.assertIsInstance(const, constant_recipe.ConstantRecipe)
        self.assertEqual(const.constant, 0.5)
        # The 3rd mul consumes the constant on its first port 'a'
        edge = recipe.edges[edge_models.TargetHandle(node="mul_2", port="a")]
        self.assertEqual(
            edge, edge_models.SourceHandle(node="constant_0", port="constant")
        )

    def test_keyword_literal_injects_constant(self):
        recipe = workflow_parser.parse_workflow(keyword_constant)
        self.assertEqual(recipe.nodes["constant_0"].constant, 3)
        edge = recipe.edges[edge_models.TargetHandle(node="increment_0", port="step")]
        self.assertEqual(edge.node, "constant_0")

    def test_compound_literal(self):
        recipe = workflow_parser.parse_workflow(compound_constant)
        self.assertEqual(
            recipe.nodes["constant_0"].constant, [1.0, 1, "1", {"key": [42]}]
        )

    def test_tuple_arg_rejected(self):
        with self.assertRaises(Exception) as ctx:
            workflow_parser.parse_workflow(tuple_arg)
        self.assertIn("mul_0", str(ctx.exception))

    def test_lambda_arg_rejected(self):
        with self.assertRaises(TypeError):
            workflow_parser.parse_workflow(lambda_arg)


class TestMakeConstant(unittest.TestCase):
    def test_builds_recipe(self):
        from flowrep.parsers import constant_parser

        c = constant_parser.make_constant(0.5, "ctx")
        self.assertEqual(c.constant, 0.5)

    def test_wraps_validation_error_with_context(self):
        from flowrep.parsers import constant_parser

        with self.assertRaises(constant_parser.ConstantParseError) as ctx:
            constant_parser.make_constant((1, 2), "while assigning to 'x'")
        self.assertIn("while assigning to 'x'", str(ctx.exception))


class TestConsumeSource(unittest.TestCase):
    def test_consume_source_creates_peer_edge(self):
        scope = symbol_scope.SymbolScope({})
        source = edge_models.SourceHandle(node="constant_0", port="constant")
        scope.consume_source(source, "consumer", "a")
        edges = scope.edges
        self.assertEqual(
            edges[edge_models.TargetHandle(node="consumer", port="a")], source
        )


if __name__ == "__main__":
    unittest.main()
