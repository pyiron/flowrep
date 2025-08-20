import unittest

from flowrep.datastructure import _HasToDictionaryMapping


class ConcreteHtDM(_HasToDictionaryMapping[int]): ...


class TestHasToDictionaryMapping(unittest.TestCase):
    def test_mapping(self):
        t = (1, 2, 3)
        a, b, c = t
        mapping = ConcreteHtDM(a=a, b=b)
        self.assertEqual(mapping["a"], a)
        self.assertEqual(mapping["b"], b)

        mapping["c"] = c
        self.assertEqual(mapping.c, c)

        self.assertEqual(len(mapping), len(t))

        del mapping["b"]
        self.assertEqual(len(mapping), len(t) - 1)
        self.assertIsNone(mapping.get("b", None))
        self.assertEqual(mapping.a, a)
        self.assertEqual(mapping.c, c)


if __name__ == "__main__":
    unittest.main()
