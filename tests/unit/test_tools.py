import unittest

from flowrep import tools


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


if __name__ == "__main__":
    unittest.main()
