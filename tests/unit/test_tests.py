import unittest

import flowrep


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = flowrep.__version__
        print(version)
        self.assertTrue(version.startswith("0"))
