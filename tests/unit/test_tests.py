import unittest

import chefflow


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = chefflow.__version__
        print(version)
        self.assertTrue(version.startswith("0"))
