import unittest

import pydantic
from pyiron_snippets import versions

from flowrep.models import base_models
from flowrep.models.nodes import (
    atomic_model,
    helper_models,
    workflow_model,
)

from flowrep_static import test_helpers


class TestConditionalCaseValidation(unittest.TestCase):
    @staticmethod
    def _make_condition(outputs=None):
        return helper_models.LabeledNode(
            label="condition",
            node=test_helpers.make_atomic(
                inputs=["x"], outputs=outputs or ["result"], qualname="check"
            ),
        )

    @staticmethod
    def _make_body():
        return helper_models.LabeledNode(
            label="body",
            node=test_helpers.make_atomic(
                inputs=["x"], outputs=["y"], qualname="handle"
            ),
        )

    def test_single_output_condition_no_explicit_output(self):
        """ConditionalCase with single-output condition needs no condition_output."""
        case = helper_models.ConditionalCase(
            condition=self._make_condition(outputs=["result"]),
            body=self._make_body(),
        )
        self.assertIsNone(case.condition_output)

    def test_multi_output_condition_requires_explicit_output(self):
        """ConditionalCase with multi-output condition must specify condition_output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            helper_models.ConditionalCase(
                condition=self._make_condition(outputs=["a", "b"]),
                body=self._make_body(),
            )
        self.assertIn("exactly one output", str(ctx.exception))

    def test_multi_output_condition_with_valid_output(self):
        """ConditionalCase with explicit valid condition_output should pass."""
        case = helper_models.ConditionalCase(
            condition=self._make_condition(outputs=["a", "b"]),
            body=self._make_body(),
            condition_output="a",
        )
        self.assertEqual(case.condition_output, "a")

    def test_invalid_condition_output_rejected(self):
        """ConditionalCase with invalid condition_output should fail."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            helper_models.ConditionalCase(
                condition=self._make_condition(outputs=["a", "b"]),
                body=self._make_body(),
                condition_output="nonexistent",
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestExceptionCaseValidation(unittest.TestCase):
    @staticmethod
    def _make_except_body(inputs=None, outputs=None) -> atomic_model.AtomicNode:
        return atomic_model.AtomicNode(
            reference=test_helpers.make_reference("mod", "handle_error"),
            inputs=inputs or ["x"],
            outputs=outputs or ["y"],
        )

    def test_valid_single_exception(self):
        """ExceptionCase with a single exception type should validate."""
        value_info = versions.VersionInfo.of(ValueError)
        case = helper_models.ExceptionCase(
            exceptions=[value_info],
            body=helper_models.LabeledNode(
                label="handler", node=self._make_except_body()
            ),
        )
        self.assertEqual(case.exceptions, [value_info])

    def test_valid_multiple_exceptions(self):
        """ExceptionCase with multiple exception types should validate."""
        case = helper_models.ExceptionCase(
            exceptions=[
                versions.VersionInfo.of(ValueError),
                versions.VersionInfo.of(TypeError),
                versions.VersionInfo.of(KeyError),
            ],
            body=helper_models.LabeledNode(
                label="handler", node=self._make_except_body()
            ),
        )
        self.assertEqual(len(case.exceptions), 3)

    def test_empty_exceptions_rejected(self):
        """ExceptionCase must have at least one exception type."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            helper_models.ExceptionCase(
                exceptions=[],
                body=helper_models.LabeledNode(
                    label="handler", node=self._make_except_body()
                ),
            )
        self.assertIn("at least one", str(ctx.exception))


class TestExceptionCaseSerialization(unittest.TestCase):
    def test_exception_case_roundtrip(self):
        """ExceptionCase JSON roundtrip."""
        original = helper_models.ExceptionCase(
            exceptions=[
                versions.VersionInfo.of(ValueError),
                versions.VersionInfo.of(TypeError),
            ],
            body=helper_models.LabeledNode(
                label="handler",
                node=atomic_model.AtomicNode(
                    reference=test_helpers.make_reference("mod", "handle_error"),
                    inputs=["x"],
                    outputs=["y"],
                ),
            ),
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = helper_models.ExceptionCase.model_validate(data)
                self.assertEqual(len(restored.exceptions), 2)
                self.assertEqual(restored.body.label, "handler")


class TestLabeledNode(unittest.TestCase):
    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        helper_models.LabeledNode.model_json_schema()

    def test_valid_labeled_node(self):
        """LabeledNode with valid label and node."""
        ln = helper_models.LabeledNode(
            label="my_node",
            node=test_helpers.make_atomic(inputs=["x"], outputs=["y"]),
        )
        self.assertEqual(ln.label, "my_node")
        self.assertIsInstance(ln.node, atomic_model.AtomicNode)

    def test_labeled_node_with_workflow(self):
        """LabeledNode can contain a WorkflowNode."""
        inner = workflow_model.WorkflowNode(
            inputs=["a"],
            outputs=["b"],
            nodes={"leaf": test_helpers.make_atomic(inputs=["x"], outputs=["y"])},
            input_edges={"leaf.x": "a"},
            edges={},
            output_edges={"b": "leaf.y"},
        )
        ln = helper_models.LabeledNode(label="nested", node=inner)
        self.assertIsInstance(ln.node, workflow_model.WorkflowNode)

    def test_invalid_label_keyword(self):
        """LabeledNode rejects Python keywords as labels."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            helper_models.LabeledNode(label="for", node=test_helpers.make_atomic())
        self.assertIn("valid Python identifier", str(ctx.exception))

    def test_invalid_label_reserved(self):
        """LabeledNode rejects reserved names as labels."""
        for reserved in base_models.RESERVED_NAMES:
            with (
                self.subTest(label=reserved),
                self.assertRaises(pydantic.ValidationError),
            ):
                helper_models.LabeledNode(
                    label=reserved, node=test_helpers.make_atomic()
                )

    def test_invalid_label_not_identifier(self):
        """LabeledNode rejects non-identifiers as labels."""
        for invalid in ["1bad", "my-label", "has space", ""]:
            with (
                self.subTest(label=invalid),
                self.assertRaises(pydantic.ValidationError),
            ):
                helper_models.LabeledNode(
                    label=invalid, node=test_helpers.make_atomic()
                )


class TestConditionalCase(unittest.TestCase):
    """Tests for ConditionalCase validation."""

    def test_valid_single_output_inferred(self):
        """condition_output inferred when condition has exactly one output."""
        cc = helper_models.ConditionalCase(
            condition=helper_models.LabeledNode(
                label="cond",
                node=test_helpers.make_atomic(inputs=["x"], outputs=["result"]),
            ),
            body=helper_models.LabeledNode(
                label="body",
                node=test_helpers.make_atomic(inputs=["y"], outputs=["out"]),
            ),
        )
        self.assertIsNone(cc.condition_output)

    def test_valid_explicit_condition_output(self):
        """condition_output explicitly specified."""
        cc = helper_models.ConditionalCase(
            condition=helper_models.LabeledNode(
                label="cond",
                node=test_helpers.make_atomic(inputs=["x"], outputs=["a", "b", "flag"]),
            ),
            body=helper_models.LabeledNode(
                label="body",
                node=test_helpers.make_atomic(inputs=["y"], outputs=["out"]),
            ),
            condition_output="flag",
        )
        self.assertEqual(cc.condition_output, "flag")

    def test_multiple_outputs_without_condition_output_rejected(self):
        """Multiple condition outputs require explicit condition_output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="cond",
                    node=test_helpers.make_atomic(inputs=["x"], outputs=["a", "b"]),
                ),
                body=helper_models.LabeledNode(
                    label="body",
                    node=test_helpers.make_atomic(inputs=["y"], outputs=["out"]),
                ),
            )
        self.assertIn("exactly one output", str(ctx.exception))

    def test_condition_output_not_in_outputs_rejected(self):
        """condition_output must exist in condition node outputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="cond",
                    node=test_helpers.make_atomic(inputs=["x"], outputs=["a", "b"]),
                ),
                body=helper_models.LabeledNode(
                    label="body",
                    node=test_helpers.make_atomic(inputs=["y"], outputs=["out"]),
                ),
                condition_output="nonexistent",
            )
        self.assertIn("not found among available outputs", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_distinct_labels_valid(self):
        """Condition and body can have different labels."""
        cc = helper_models.ConditionalCase(
            condition=helper_models.LabeledNode(
                label="check", node=test_helpers.make_atomic(outputs=["ok"])
            ),
            body=helper_models.LabeledNode(
                label="run", node=test_helpers.make_atomic()
            ),
        )
        self.assertEqual(cc.condition.label, "check")
        self.assertEqual(cc.body.label, "run")

    def test_same_labels_rejected(self):
        """Condition and body must have distinct labels."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="same", node=test_helpers.make_atomic(outputs=["ok"])
                ),
                body=helper_models.LabeledNode(
                    label="same", node=test_helpers.make_atomic()
                ),
            )
        self.assertIn("distinct labels", str(ctx.exception))
        self.assertIn("same", str(ctx.exception))
