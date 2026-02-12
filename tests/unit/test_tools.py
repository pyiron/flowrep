import unittest

from flowrep import tools


def example_function(x):
    """An example function to be hashed."""
    return x * 2


class TestTools(unittest.TestCase):
    def test_defaultdict(self):
        d = tools.recursive_defaultdict()
        d["x"]["y"]["z"] = 3
        self.assertEqual(d["x"]["y"]["z"], 3)
        normal = {"a": {"b": {"c": 1}}}
        dd = tools.dict_to_recursive_dd(normal)
        self.assertEqual(dd["a"]["b"]["c"], 1)
        dd["x"]["y"]["z"] = 2
        self.assertEqual(dd["x"]["y"]["z"], 2)
        plain = tools.recursive_dd_to_dict(dd)
        self.assertEqual(plain["a"]["b"]["c"], 1)

    def test_get_function_metadata(self):
        meta = tools.get_function_metadata(example_function, full_metadata=True)
        self.assertIn("name", meta)
        self.assertIn("module", meta)
        self.assertIn("docstring", meta)
        self.assertEqual(meta["name"], "example_function")
        meta = tools.get_function_metadata(example_function, full_metadata=False)
        self.assertNotIn("docstring", meta)

    def test_hash_function(self):
        import math

        self.assertEqual(
            tools.hash_function(math.sin)[:40],
            "sin:c5f6a0370fc318866315717568a5c3d2d704",
        )

        def function_with_list():
            return [1, 2, 3]

        self.assertEqual(
            tools._hash_function_with_ast(function_with_list)[:40],
            "function_with_list:3655ea86940164b32b89e",
            msg="AST based hash should be OS and Python version independent",
        )
        self.assertEqual(
            tools._hash_function_with_ast(example_function)[:40],
            "example_function:853fe8c9763d182350fa903",
            msg="AST based hash should be OS and Python version independent",
        )


if __name__ == "__main__":
    unittest.main()
