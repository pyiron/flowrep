import math
import unittest

from flowrep import crawler


def add(x, y):
    return x + y


def op(a, b):
    c = add(a, b)
    d = math.sqrt(c)
    return d


def more_op(a, b):
    c = op(a, b)
    return c


class TestCrawler(unittest.TestCase):
    def test_analyze_function_dependencies(self):
        loc, ext = crawler.analyze_function_dependencies(op)
        self.assertEqual(loc, {add})
        self.assertEqual(len(ext), 1)
        f = ext.pop()
        self.assertEqual(f.fully_qualified_name, "math.sqrt")
        loc, ext = crawler.analyze_function_dependencies(more_op)
        self.assertEqual(loc, {op, add})
        self.assertEqual(len(ext), 1)
        g = ext.pop()
        self.assertEqual(g.fully_qualified_name, "math.sqrt")


if __name__ == "__main__":
    unittest.main()
