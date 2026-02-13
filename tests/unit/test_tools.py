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

        # Test built-in function (triggers except path - no source code available)
        self.assertEqual(
            tools.hash_function(math.sin)[:40],
            "sin:0146c21ab456a735f07d62b456f003ce3dc6",
        )

        # Test regular function with source code available
        expected_hash = "example_function:196938631e98c05e128b0b1"
        self.assertEqual(
            tools.hash_function(example_function)[: len(expected_hash)],
            expected_hash,
        )

    def test_hash_function_fallback(self):
        """Test that hash_function falls back to signature for functions without source."""
        import math

        # Built-in functions should not raise exceptions
        hash_result = tools.hash_function(math.sin)
        self.assertTrue(hash_result.startswith("sin:"))
        self.assertEqual(len(hash_result), 68)  # "sin:" + 64 char hex hash

        # Test another built-in
        hash_result = tools.hash_function(len)
        self.assertTrue(hash_result.startswith("len:"))
        self.assertEqual(len(hash_result), 68)


if __name__ == "__main__":
    unittest.main()
