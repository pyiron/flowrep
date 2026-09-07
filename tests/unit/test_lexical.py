import unittest

from flowrep import base_models, edge_models, lexical


class TestLexical(unittest.TestCase):
    def test_delimiter_matches_handle_model(self):
        self.assertEqual(lexical.DELIMITER, edge_models.HandleModel.delimiter)

    def test_join(self):
        self.assertEqual(lexical.join("sub", "child"), "sub.child")

    def test_join_skips_empty_segments(self):
        """The drawing root has an empty path; joining onto it must not lead with a dot."""
        self.assertEqual(lexical.join("", "child"), "child")

    def test_split(self):
        self.assertEqual(lexical.split("sub.child"), ("sub", "child"))

    def test_split_empty_is_empty(self):
        self.assertEqual(lexical.split(""), ())

    def test_round_trip(self):
        for path in ("", "a", "a.b", "a.b.c"):
            with self.subTest(path=path):
                self.assertEqual(lexical.join(*lexical.split(path)), path)

    def test_port_path(self):
        self.assertEqual(
            lexical.port_path("sub.body_0", base_models.IOTypes.INPUTS, "x"),
            "sub.body_0.inputs.x",
        )

    def test_port_path_on_root(self):
        self.assertEqual(
            lexical.port_path("", base_models.IOTypes.OUTPUTS, "y"),
            "outputs.y",
        )
