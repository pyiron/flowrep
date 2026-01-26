import unittest

import pydantic

from flowrep.models import base_models, edge_models, subgraph_protocols
from flowrep.models.nodes import (
    atomic_model,
    helper_models,
    try_model,
    union,
    workflow_model,
)


def _make_try_body(inputs=None, outputs=None) -> atomic_model.AtomicNode:
    return atomic_model.AtomicNode(
        fully_qualified_name="mod.try_func",
        inputs=inputs or ["x"],
        outputs=outputs or ["y"],
    )


def _make_except_body(inputs=None, outputs=None) -> atomic_model.AtomicNode:
    return atomic_model.AtomicNode(
        fully_qualified_name="mod.handle_error",
        inputs=inputs or ["x"],
        outputs=outputs or ["y"],
    )


def _make_exception_case(
    n: int,
    exceptions: list[str] | None = None,
    inputs=None,
    outputs=None,
) -> helper_models.ExceptionCase:
    return helper_models.ExceptionCase(
        exceptions=exceptions or ["builtins.ValueError"],
        body=helper_models.LabeledNode(
            label=f"except_{n}", node=_make_except_body(inputs=inputs, outputs=outputs)
        ),
    )


def _make_input_edges(try_node, exception_cases):
    edges_dict = {
        edge_models.TargetHandle(
            node=try_node.label, port="x"
        ): edge_models.InputSource(port="inp")
    }
    for case in exception_cases:
        edges_dict[edge_models.TargetHandle(node=case.body.label, port="x")] = (
            edge_models.InputSource(port="inp")
        )
    return edges_dict


def _make_prospective_output_edges(try_node, exception_cases):
    sources = [edge_models.SourceHandle(node=try_node.label, port="y")]
    for case in exception_cases:
        sources.append(edge_models.SourceHandle(node=case.body.label, port="y"))
    return {edge_models.OutputTarget(port="out"): sources}


def _make_valid_try_node(n_exception_cases=1):
    try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
    exception_cases = [_make_exception_case(n) for n in range(n_exception_cases)]

    return try_model.TryNode(
        inputs=["inp"],
        outputs=["out"],
        try_node=try_node,
        exception_cases=exception_cases,
        input_edges=_make_input_edges(try_node, exception_cases),
        prospective_output_edges=_make_prospective_output_edges(
            try_node, exception_cases
        ),
    )


class TestTryNodeBasicConstruction(unittest.TestCase):
    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        try_model.TryNode.model_json_schema()

    def test_obeys_build_subgraph_with_dynamic_output(self):
        """TryNode should obey build subgraph with dynamic output."""
        node = _make_valid_try_node()
        self.assertIsInstance(node, subgraph_protocols.BuildsSubgraphWithDynamicOutput)

    def test_valid_single_exception_case(self):
        """TryNode with one exception case should validate."""
        node = _make_valid_try_node(n_exception_cases=1)
        self.assertEqual(node.type, base_models.RecipeElementType.TRY)
        self.assertEqual(len(node.exception_cases), 1)

    def test_valid_multiple_exception_cases(self):
        """TryNode with multiple exception cases should validate."""
        node = _make_valid_try_node(n_exception_cases=3)
        self.assertEqual(len(node.exception_cases), 3)


class TestTryNodeExceptionCasesValidation(unittest.TestCase):
    def test_empty_exception_cases_rejected(self):
        """TryNode must have at least one exception case."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=[],
                input_edges={},
                prospective_output_edges={edge_models.OutputTarget(port="out"): []},
            )
        self.assertIn("at least one", str(ctx.exception))

    def test_duplicate_labels_try_and_except_rejected(self):
        """Labels must be unique between try_node and exception cases."""
        try_node = helper_models.LabeledNode(
            label="shared_label", node=_make_try_body()
        )
        exception_case = helper_models.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=helper_models.LabeledNode(
                label="shared_label", node=_make_except_body()
            ),  # Duplicate
        )
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=[exception_case],
                input_edges={},
                prospective_output_edges={
                    edge_models.OutputTarget(port="out"): [
                        edge_models.SourceHandle(node="shared_label", port="y"),
                    ]
                },
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_duplicate_labels_across_exception_cases_rejected(self):
        """Labels must be unique across exception cases."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        case0 = helper_models.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=helper_models.LabeledNode(label="handler", node=_make_except_body()),
        )
        case1 = helper_models.ExceptionCase(
            exceptions=["builtins.TypeError"],
            body=helper_models.LabeledNode(
                label="handler", node=_make_except_body()
            ),  # Dup
        )
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=[case0, case1],
                input_edges={},
                prospective_output_edges={
                    edge_models.OutputTarget(port="out"): [
                        edge_models.SourceHandle(node="try_body", port="y"),
                        edge_models.SourceHandle(node="handler", port="y"),
                    ]
                },
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_exception_cases_accept_various_node_types(self):
        """Exception case bodies can be any NodeModel type."""
        workflow_body = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "inner": atomic_model.AtomicNode(
                    fully_qualified_name="mod.f",
                    inputs=["a"],
                    outputs=["b"],
                )
            },
            input_edges={
                edge_models.TargetHandle(
                    node="inner", port="a"
                ): edge_models.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="inner", port="b"
                ),
            },
        )

        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_case = helper_models.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=helper_models.LabeledNode(
                label="workflow_handler", node=workflow_body
            ),
        )

        node = try_model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=[exception_case],
            input_edges=_make_input_edges(try_node, [exception_case]),
            prospective_output_edges=_make_prospective_output_edges(
                try_node, [exception_case]
            ),
        )
        self.assertIsInstance(
            node.exception_cases[0].body.node, workflow_model.WorkflowNode
        )


class TestTryNodeInputEdgesValidation(unittest.TestCase):
    def test_input_edges_invalid_target_node(self):
        """input_edges targets must reference existing prospective nodes."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges={
                    edge_models.TargetHandle(
                        node="nonexistent", port="x"
                    ): edge_models.InputSource(port="inp")
                },
                prospective_output_edges=_make_prospective_output_edges(
                    try_node, exception_cases
                ),
            )
        exc_str = str(ctx.exception)
        self.assertIn("nonexistent", exc_str)

    def test_input_edges_can_target_try_node(self):
        """input_edges can target the try_node."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        node = try_model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=exception_cases,
            input_edges={
                edge_models.TargetHandle(
                    node="try_body", port="x"
                ): edge_models.InputSource(port="inp"),
            },
            prospective_output_edges=_make_prospective_output_edges(
                try_node, exception_cases
            ),
        )
        self.assertEqual(len(node.input_edges), 1)

    def test_input_edges_can_target_exception_cases(self):
        """input_edges can target exception case bodies."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(n) for n in range(2)]
        node = try_model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=exception_cases,
            input_edges={
                edge_models.TargetHandle(
                    node="except_0", port="x"
                ): edge_models.InputSource(port="inp"),
                edge_models.TargetHandle(
                    node="except_1", port="x"
                ): edge_models.InputSource(port="inp"),
            },
            prospective_output_edges=_make_prospective_output_edges(
                try_node, exception_cases
            ),
        )
        self.assertEqual(len(node.input_edges), 2)

    def test_input_edges_invalid_target_port(self):
        """input_edges target port must exist on the target node."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges={
                    edge_models.TargetHandle(
                        node="try_body", port="nonexistent"
                    ): edge_models.InputSource(port="inp")
                },
                prospective_output_edges=_make_prospective_output_edges(
                    try_node, exception_cases
                ),
            )
        exc_str = str(ctx.exception)
        self.assertIn("Invalid input_edges target ports", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_input_edges_invalid_source_port(self):
        """input_edges source port must exist on the TryNode inputs."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges={
                    edge_models.TargetHandle(
                        node="try_body", port="x"
                    ): edge_models.InputSource(port="nonexistent")
                },
                prospective_output_edges=_make_prospective_output_edges(
                    try_node, exception_cases
                ),
            )
        exc_str = str(ctx.exception)
        self.assertIn("Invalid input_edges source ports", exc_str)
        self.assertIn("nonexistent", exc_str)


class TestTryNodeProspectiveOutputEdgesValidation(unittest.TestCase):
    def test_prospective_output_edges_invalid_source_node(self):
        """Sources must reference valid prospective nodes."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                prospective_output_edges={
                    edge_models.OutputTarget(port="out"): [
                        edge_models.SourceHandle(node="nonexistent", port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid", exc_str.lower())
        self.assertIn("nonexistent", exc_str)

    def test_prospective_output_edges_duplicate_source_node_rejected(self):
        """Each prospective node can appear at most once per output."""
        try_node = helper_models.LabeledNode(
            label="try_body", node=_make_try_body(outputs=["y", "z"])
        )
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                prospective_output_edges={
                    edge_models.OutputTarget(port="out"): [
                        edge_models.SourceHandle(node="try_body", port="y"),
                        edge_models.SourceHandle(
                            node="try_body", port="z"
                        ),  # Duplicate source node
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("must be unique", exc_str)
        self.assertIn("duplicate", exc_str.lower())

    def test_prospective_output_edges_keys_must_match_outputs(self):
        """prospective_output_edges keys must match TryNode outputs."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out", "other"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                prospective_output_edges={
                    edge_models.OutputTarget(port="out"): [
                        edge_models.SourceHandle(node="try_body", port="y"),
                    ]
                    # Missing "other"
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("Missing output edge for", exc_str)
        self.assertIn("other", exc_str)

    def test_prospective_output_edges_extra_key_rejected(self):
        """prospective_output_edges cannot have keys not in outputs."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                prospective_output_edges={
                    edge_models.OutputTarget(port="out"): [
                        edge_models.SourceHandle(node="try_body", port="y"),
                    ],
                    edge_models.OutputTarget(port="extra"): [
                        edge_models.SourceHandle(node="try_body", port="y"),
                    ],
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("Invalid output target ports", exc_str)
        self.assertIn("extra", exc_str)

    def test_prospective_output_edges_empty_sources_rejected(self):
        """An output must have at least one source."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                prospective_output_edges={edge_models.OutputTarget(port="out"): []},
            )
        exc_str = str(ctx.exception)
        self.assertIn("cannot be empty", exc_str)
        self.assertIn("out", exc_str)

    def test_prospective_output_edges_partial_sources_allowed(self):
        """An output can have sources from only some prospective nodes."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(n) for n in range(3)]
        node = try_model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=exception_cases,
            input_edges=_make_input_edges(try_node, exception_cases),
            prospective_output_edges={
                edge_models.OutputTarget(port="out"): [
                    # Only try_body and except_0, skipping except_1 and except_2
                    edge_models.SourceHandle(node="try_body", port="y"),
                    edge_models.SourceHandle(node="except_0", port="y"),
                ]
            },
        )
        self.assertEqual(
            len(node.prospective_output_edges[edge_models.OutputTarget(port="out")]), 2
        )

    def test_prospective_output_edges_all_sources_allowed(self):
        """An output can have sources from all prospective nodes."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(n) for n in range(2)]
        node = try_model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=exception_cases,
            input_edges=_make_input_edges(try_node, exception_cases),
            prospective_output_edges={
                edge_models.OutputTarget(port="out"): [
                    edge_models.SourceHandle(node="try_body", port="y"),
                    edge_models.SourceHandle(node="except_0", port="y"),
                    edge_models.SourceHandle(node="except_1", port="y"),
                ]
            },
        )
        self.assertEqual(
            len(node.prospective_output_edges[edge_models.OutputTarget(port="out")]), 3
        )

    def test_prospective_output_edges_invalid_source_port(self):
        """prospective_output_edges source port must exist on the source node."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                prospective_output_edges={
                    edge_models.OutputTarget(port="out"): [
                        edge_models.SourceHandle(node="try_body", port="nonexistent"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("Invalid output source ports", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_prospective_output_edges_valid_multiple_outputs(self):
        """prospective_output_edges works with multiple outputs."""
        body_node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["out1", "out2"],
        )
        try_node = helper_models.LabeledNode(label="try_body", node=body_node)
        exception_case = helper_models.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=helper_models.LabeledNode(label="except_body", node=body_node),
        )
        node = try_model.TryNode(
            inputs=["inp"],
            outputs=["a", "b"],
            try_node=try_node,
            exception_cases=[exception_case],
            input_edges={
                edge_models.TargetHandle(
                    node="try_body", port="x"
                ): edge_models.InputSource(port="inp"),
                edge_models.TargetHandle(
                    node="except_body", port="x"
                ): edge_models.InputSource(port="inp"),
            },
            prospective_output_edges={
                edge_models.OutputTarget(port="a"): [
                    edge_models.SourceHandle(node="try_body", port="out1"),
                    edge_models.SourceHandle(node="except_body", port="out1"),
                ],
                edge_models.OutputTarget(port="b"): [
                    edge_models.SourceHandle(node="try_body", port="out2"),
                    edge_models.SourceHandle(node="except_body", port="out2"),
                ],
            },
        )
        self.assertEqual(len(node.prospective_output_edges), 2)

    def test_empty_outputs_with_empty_matrix(self):
        """TryNode with no outputs requires empty prospective_output_edges."""
        try_node = helper_models.LabeledNode(
            label="try_body",
            node=atomic_model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=["x"],
                outputs=[],
            ),
        )
        exception_case = helper_models.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=helper_models.LabeledNode(
                label="handler",
                node=atomic_model.AtomicNode(
                    fully_qualified_name="mod.handler",
                    inputs=["x"],
                    outputs=[],
                ),
            ),
        )
        node = try_model.TryNode(
            inputs=["inp"],
            outputs=[],
            try_node=try_node,
            exception_cases=[exception_case],
            input_edges={
                edge_models.TargetHandle(
                    node="try_body", port="x"
                ): edge_models.InputSource(port="inp"),
            },
            prospective_output_edges={},
        )
        self.assertEqual(node.outputs, [])
        self.assertEqual(node.prospective_output_edges, {})


class TestTryNodeProspectiveNodes(unittest.TestCase):
    def test_prospective_nodes_single_exception_case(self):
        """prospective_nodes includes try_node and exception case bodies."""
        node = _make_valid_try_node(n_exception_cases=1)
        prospective = node.prospective_nodes
        self.assertIn("try_body", prospective)
        self.assertIn("except_0", prospective)
        self.assertEqual(len(prospective), 2)

    def test_prospective_nodes_multiple_exception_cases(self):
        """prospective_nodes includes all exception case bodies."""
        node = _make_valid_try_node(n_exception_cases=3)
        prospective = node.prospective_nodes
        self.assertIn("try_body", prospective)
        self.assertIn("except_0", prospective)
        self.assertIn("except_1", prospective)
        self.assertIn("except_2", prospective)
        self.assertEqual(len(prospective), 4)

    def test_prospective_nodes_conflicting_labels_rejected(self):
        """prospective_nodes rejects nodes with conflicting labels."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_case = helper_models.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=helper_models.LabeledNode(label="try_body", node=_make_except_body()),
        )
        with self.assertRaises(pydantic.ValidationError) as ctx:
            try_model.TryNode(
                inputs=["inp"],
                outputs=[],
                try_node=try_node,
                exception_cases=[exception_case],
                input_edges={},
                prospective_output_edges={},
            )
        ctx_str = str(ctx.exception)
        self.assertIn("must have unique elements", ctx_str)
        self.assertIn("Duplicates", ctx_str)


class TestTryNodeSerialization(unittest.TestCase):
    def test_roundtrip(self):
        """Serialization roundtrip."""
        original = _make_valid_try_node(n_exception_cases=2)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = try_model.TryNode.model_validate(data)
                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(
                    len(original.exception_cases), len(restored.exception_cases)
                )
                self.assertEqual(original.type, restored.type)

    def test_roundtrip_multiple_exception_types(self):
        """Roundtrip with multiple exception types per case."""
        try_node = helper_models.LabeledNode(label="try_body", node=_make_try_body())
        exception_case = helper_models.ExceptionCase(
            exceptions=[
                "builtins.ValueError",
                "builtins.TypeError",
                "builtins.KeyError",
            ],
            body=helper_models.LabeledNode(label="handler", node=_make_except_body()),
        )
        original = try_model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=[exception_case],
            input_edges=_make_input_edges(try_node, [exception_case]),
            prospective_output_edges=_make_prospective_output_edges(
                try_node, [exception_case]
            ),
        )

        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = try_model.TryNode.model_validate(data)
                self.assertEqual(len(restored.exception_cases[0].exceptions), 3)

    def test_discriminated_union_roundtrip(self):
        """Ensure type discriminator works for polymorphic deserialization."""
        original = _make_valid_try_node()
        data = original.model_dump(mode="json")

        node = pydantic.TypeAdapter(union.NodeType).validate_python(data)
        self.assertIsInstance(node, try_model.TryNode)


class TestTryNodeInWorkflow(unittest.TestCase):
    def test_try_node_as_workflow_child(self):
        """TryNode can be used as a child node in a WorkflowNode."""
        try_node = _make_valid_try_node()
        workflow = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"try_block": try_node},
            input_edges={
                edge_models.TargetHandle(
                    node="try_block", port="inp"
                ): edge_models.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="try_block", port="out"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["try_block"], try_model.TryNode)

    def test_try_node_multiple_exception_cases_as_workflow_child(self):
        """TryNode with multiple exception cases can be a workflow child."""
        try_node = _make_valid_try_node(n_exception_cases=3)
        workflow = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"try_block": try_node},
            input_edges={
                edge_models.TargetHandle(
                    node="try_block", port="inp"
                ): edge_models.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="try_block", port="out"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["try_block"], try_model.TryNode)
        self.assertEqual(len(workflow.nodes["try_block"].exception_cases), 3)

    def test_workflow_with_try_node_roundtrip(self):
        """Workflow containing TryNode serializes and deserializes correctly."""
        try_node = _make_valid_try_node(n_exception_cases=2)
        original = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"try_block": try_node},
            input_edges={
                edge_models.TargetHandle(
                    node="try_block", port="inp"
                ): edge_models.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="try_block", port="out"
                ),
            },
        )

        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = workflow_model.WorkflowNode.model_validate(data)
                self.assertIsInstance(restored.nodes["try_block"], try_model.TryNode)
                self.assertEqual(len(restored.nodes["try_block"].exception_cases), 2)


if __name__ == "__main__":
    unittest.main()
