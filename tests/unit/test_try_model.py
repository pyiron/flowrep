import unittest

import pydantic

from flowrep import model


def _make_try_body(inputs=None, outputs=None) -> model.AtomicNode:
    return model.AtomicNode(
        fully_qualified_name="mod.try_func",
        inputs=inputs or ["x"],
        outputs=outputs or ["y"],
    )


def _make_except_body(inputs=None, outputs=None) -> model.AtomicNode:
    return model.AtomicNode(
        fully_qualified_name="mod.handle_error",
        inputs=inputs or ["x"],
        outputs=outputs or ["y"],
    )


def _make_exception_case(
    n: int,
    exceptions: list[str] | None = None,
    inputs=None,
    outputs=None,
) -> model.ExceptionCase:
    return model.ExceptionCase(
        exceptions=exceptions or ["builtins.ValueError"],
        body=model.LabeledNode(
            label=f"except_{n}", node=_make_except_body(inputs=inputs, outputs=outputs)
        ),
    )


def _make_input_edges(try_node, exception_cases):
    edges = {
        model.TargetHandle(node=try_node.label, port="x"): model.InputSource(port="inp")
    }
    for case in exception_cases:
        edges[model.TargetHandle(node=case.body.label, port="x")] = model.InputSource(
            port="inp"
        )
    return edges


def _make_output_edges_matrix(try_node, exception_cases):
    sources = [model.SourceHandle(node=try_node.label, port="y")]
    for case in exception_cases:
        sources.append(model.SourceHandle(node=case.body.label, port="y"))
    return {model.OutputTarget(port="out"): sources}


def _make_valid_try_node(n_exception_cases=1):
    try_node = model.LabeledNode(label="try_body", node=_make_try_body())
    exception_cases = [_make_exception_case(n) for n in range(n_exception_cases)]

    return model.TryNode(
        inputs=["inp"],
        outputs=["out"],
        try_node=try_node,
        exception_cases=exception_cases,
        input_edges=_make_input_edges(try_node, exception_cases),
        output_edges_matrix=_make_output_edges_matrix(try_node, exception_cases),
    )


class TestExceptionCaseValidation(unittest.TestCase):
    def test_valid_single_exception(self):
        """ExceptionCase with a single exception type should validate."""
        case = model.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=model.LabeledNode(label="handler", node=_make_except_body()),
        )
        self.assertEqual(case.exceptions, ["builtins.ValueError"])

    def test_valid_multiple_exceptions(self):
        """ExceptionCase with multiple exception types should validate."""
        case = model.ExceptionCase(
            exceptions=[
                "builtins.ValueError",
                "builtins.TypeError",
                "builtins.KeyError",
            ],
            body=model.LabeledNode(label="handler", node=_make_except_body()),
        )
        self.assertEqual(len(case.exceptions), 3)

    def test_empty_exceptions_rejected(self):
        """ExceptionCase must have at least one exception type."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ExceptionCase(
                exceptions=[],
                body=model.LabeledNode(label="handler", node=_make_except_body()),
            )
        self.assertIn("at least one", str(ctx.exception))


class TestTryNodeBasicConstruction(unittest.TestCase):
    def test_valid_single_exception_case(self):
        """TryNode with one exception case should validate."""
        node = _make_valid_try_node(n_exception_cases=1)
        self.assertEqual(node.type, model.RecipeElementType.TRY)
        self.assertEqual(len(node.exception_cases), 1)

    def test_valid_multiple_exception_cases(self):
        """TryNode with multiple exception cases should validate."""
        node = _make_valid_try_node(n_exception_cases=3)
        self.assertEqual(len(node.exception_cases), 3)


class TestTryNodeExceptionCasesValidation(unittest.TestCase):
    def test_empty_exception_cases_rejected(self):
        """TryNode must have at least one exception case."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=[],
                input_edges={},
                output_edges_matrix={model.OutputTarget(port="out"): []},
            )
        self.assertIn("at least one", str(ctx.exception))

    def test_duplicate_labels_try_and_except_rejected(self):
        """Labels must be unique between try_node and exception cases."""
        try_node = model.LabeledNode(label="shared_label", node=_make_try_body())
        exception_case = model.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=model.LabeledNode(
                label="shared_label", node=_make_except_body()
            ),  # Duplicate
        )
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=[exception_case],
                input_edges={},
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node="shared_label", port="y"),
                    ]
                },
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_duplicate_labels_across_exception_cases_rejected(self):
        """Labels must be unique across exception cases."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        case0 = model.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=model.LabeledNode(label="handler", node=_make_except_body()),
        )
        case1 = model.ExceptionCase(
            exceptions=["builtins.TypeError"],
            body=model.LabeledNode(label="handler", node=_make_except_body()),  # Dup
        )
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=[case0, case1],
                input_edges={},
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node="try_body", port="y"),
                        model.SourceHandle(node="handler", port="y"),
                    ]
                },
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_exception_cases_accept_various_node_types(self):
        """Exception case bodies can be any NodeModel type."""
        workflow_body = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "inner": model.AtomicNode(
                    fully_qualified_name="mod.f",
                    inputs=["a"],
                    outputs=["b"],
                )
            },
            input_edges={
                model.TargetHandle(node="inner", port="a"): model.InputSource(port="x"),
            },
            edges={},
            output_edges={
                model.OutputTarget(port="y"): model.SourceHandle(
                    node="inner", port="b"
                ),
            },
        )

        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_case = model.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=model.LabeledNode(label="workflow_handler", node=workflow_body),
        )

        node = model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=[exception_case],
            input_edges=_make_input_edges(try_node, [exception_case]),
            output_edges_matrix=_make_output_edges_matrix(try_node, [exception_case]),
        )
        self.assertIsInstance(node.exception_cases[0].body.node, model.WorkflowNode)


class TestTryNodeInputEdgesValidation(unittest.TestCase):
    def test_input_edges_invalid_target_node(self):
        """input_edges targets must reference existing prospective nodes."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges={
                    model.TargetHandle(node="nonexistent", port="x"): model.InputSource(
                        port="inp"
                    )
                },
                output_edges_matrix=_make_output_edges_matrix(
                    try_node, exception_cases
                ),
            )
        exc_str = str(ctx.exception)
        self.assertIn("nonexistent", exc_str)

    def test_input_edges_can_target_try_node(self):
        """input_edges can target the try_node."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        node = model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=exception_cases,
            input_edges={
                model.TargetHandle(node="try_body", port="x"): model.InputSource(
                    port="inp"
                ),
            },
            output_edges_matrix=_make_output_edges_matrix(try_node, exception_cases),
        )
        self.assertEqual(len(node.input_edges), 1)

    def test_input_edges_can_target_exception_cases(self):
        """input_edges can target exception case bodies."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(n) for n in range(2)]
        node = model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=exception_cases,
            input_edges={
                model.TargetHandle(node="except_0", port="x"): model.InputSource(
                    port="inp"
                ),
                model.TargetHandle(node="except_1", port="x"): model.InputSource(
                    port="inp"
                ),
            },
            output_edges_matrix=_make_output_edges_matrix(try_node, exception_cases),
        )
        self.assertEqual(len(node.input_edges), 2)

    def test_input_edges_invalid_target_port(self):
        """input_edges target port must exist on the target node."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges={
                    model.TargetHandle(
                        node="try_body", port="nonexistent"
                    ): model.InputSource(port="inp")
                },
                output_edges_matrix=_make_output_edges_matrix(
                    try_node, exception_cases
                ),
            )
        exc_str = str(ctx.exception)
        self.assertIn("has no input port", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_input_edges_invalid_source_port(self):
        """input_edges source port must exist on the TryNode inputs."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges={
                    model.TargetHandle(node="try_body", port="x"): model.InputSource(
                        port="nonexistent"
                    )
                },
                output_edges_matrix=_make_output_edges_matrix(
                    try_node, exception_cases
                ),
            )
        exc_str = str(ctx.exception)
        self.assertIn("not a TryNode input", exc_str)
        self.assertIn("nonexistent", exc_str)


class TestTryNodeOutputEdgesMatrixValidation(unittest.TestCase):
    def test_output_edges_matrix_invalid_source_node(self):
        """Sources must reference valid prospective nodes."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node="nonexistent", port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid", exc_str.lower())
        self.assertIn("nonexistent", exc_str)

    def test_output_edges_matrix_duplicate_source_node_rejected(self):
        """Each prospective node can appear at most once per output."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node="try_body", port="y"),
                        model.SourceHandle(node="try_body", port="y"),  # Duplicate
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("at most one", exc_str)
        self.assertIn("duplicates", exc_str.lower())

    def test_output_edges_matrix_keys_must_match_outputs(self):
        """output_edges_matrix keys must match TryNode outputs."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out", "other"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node="try_body", port="y"),
                    ]
                    # Missing "other"
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("must match outputs", exc_str)
        self.assertIn("other", exc_str)

    def test_output_edges_matrix_extra_key_rejected(self):
        """output_edges_matrix cannot have keys not in outputs."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node="try_body", port="y"),
                    ],
                    model.OutputTarget(port="extra"): [
                        model.SourceHandle(node="try_body", port="y"),
                    ],
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("must match outputs", exc_str)
        self.assertIn("extra", exc_str)

    def test_output_edges_matrix_empty_sources_rejected(self):
        """An output must have at least one source."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                output_edges_matrix={model.OutputTarget(port="out"): []},
            )
        exc_str = str(ctx.exception)
        self.assertIn("at least one", exc_str)

    def test_output_edges_matrix_partial_sources_allowed(self):
        """An output can have sources from only some prospective nodes."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(n) for n in range(3)]
        node = model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=exception_cases,
            input_edges=_make_input_edges(try_node, exception_cases),
            output_edges_matrix={
                model.OutputTarget(port="out"): [
                    # Only try_body and except_0, skipping except_1 and except_2
                    model.SourceHandle(node="try_body", port="y"),
                    model.SourceHandle(node="except_0", port="y"),
                ]
            },
        )
        self.assertEqual(
            len(node.output_edges_matrix[model.OutputTarget(port="out")]), 2
        )

    def test_output_edges_matrix_all_sources_allowed(self):
        """An output can have sources from all prospective nodes."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(n) for n in range(2)]
        node = model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=exception_cases,
            input_edges=_make_input_edges(try_node, exception_cases),
            output_edges_matrix={
                model.OutputTarget(port="out"): [
                    model.SourceHandle(node="try_body", port="y"),
                    model.SourceHandle(node="except_0", port="y"),
                    model.SourceHandle(node="except_1", port="y"),
                ]
            },
        )
        self.assertEqual(
            len(node.output_edges_matrix[model.OutputTarget(port="out")]), 3
        )

    def test_output_edges_matrix_invalid_source_port(self):
        """output_edges_matrix source port must exist on the source node."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_cases = [_make_exception_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.TryNode(
                inputs=["inp"],
                outputs=["out"],
                try_node=try_node,
                exception_cases=exception_cases,
                input_edges=_make_input_edges(try_node, exception_cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node="try_body", port="nonexistent"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("has no output port", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_output_edges_matrix_valid_multiple_outputs(self):
        """output_edges_matrix works with multiple outputs."""
        body_node = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["out1", "out2"],
        )
        try_node = model.LabeledNode(label="try_body", node=body_node)
        exception_case = model.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=model.LabeledNode(label="except_body", node=body_node),
        )
        node = model.TryNode(
            inputs=["inp"],
            outputs=["a", "b"],
            try_node=try_node,
            exception_cases=[exception_case],
            input_edges={
                model.TargetHandle(node="try_body", port="x"): model.InputSource(
                    port="inp"
                ),
                model.TargetHandle(node="except_body", port="x"): model.InputSource(
                    port="inp"
                ),
            },
            output_edges_matrix={
                model.OutputTarget(port="a"): [
                    model.SourceHandle(node="try_body", port="out1"),
                    model.SourceHandle(node="except_body", port="out1"),
                ],
                model.OutputTarget(port="b"): [
                    model.SourceHandle(node="try_body", port="out2"),
                    model.SourceHandle(node="except_body", port="out2"),
                ],
            },
        )
        self.assertEqual(len(node.output_edges_matrix), 2)

    def test_empty_outputs_with_empty_matrix(self):
        """TryNode with no outputs requires empty output_edges_matrix."""
        try_node = model.LabeledNode(
            label="try_body",
            node=model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=["x"],
                outputs=[],
            ),
        )
        exception_case = model.ExceptionCase(
            exceptions=["builtins.ValueError"],
            body=model.LabeledNode(
                label="handler",
                node=model.AtomicNode(
                    fully_qualified_name="mod.handler",
                    inputs=["x"],
                    outputs=[],
                ),
            ),
        )
        node = model.TryNode(
            inputs=["inp"],
            outputs=[],
            try_node=try_node,
            exception_cases=[exception_case],
            input_edges={
                model.TargetHandle(node="try_body", port="x"): model.InputSource(
                    port="inp"
                ),
            },
            output_edges_matrix={},
        )
        self.assertEqual(node.outputs, [])
        self.assertEqual(node.output_edges_matrix, {})


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


class TestTryNodeSerialization(unittest.TestCase):
    def test_roundtrip(self):
        """Serialization roundtrip."""
        original = _make_valid_try_node(n_exception_cases=2)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = model.TryNode.model_validate(data)
                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(
                    len(original.exception_cases), len(restored.exception_cases)
                )
                self.assertEqual(original.type, restored.type)

    def test_roundtrip_multiple_exception_types(self):
        """Roundtrip with multiple exception types per case."""
        try_node = model.LabeledNode(label="try_body", node=_make_try_body())
        exception_case = model.ExceptionCase(
            exceptions=[
                "builtins.ValueError",
                "builtins.TypeError",
                "builtins.KeyError",
            ],
            body=model.LabeledNode(label="handler", node=_make_except_body()),
        )
        original = model.TryNode(
            inputs=["inp"],
            outputs=["out"],
            try_node=try_node,
            exception_cases=[exception_case],
            input_edges=_make_input_edges(try_node, [exception_case]),
            output_edges_matrix=_make_output_edges_matrix(try_node, [exception_case]),
        )

        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = model.TryNode.model_validate(data)
                self.assertEqual(len(restored.exception_cases[0].exceptions), 3)

    def test_discriminated_union_roundtrip(self):
        """Ensure type discriminator works for polymorphic deserialization."""
        original = _make_valid_try_node()
        data = original.model_dump(mode="json")

        node = pydantic.TypeAdapter(model.NodeType).validate_python(data)
        self.assertIsInstance(node, model.TryNode)


class TestTryNodeInWorkflow(unittest.TestCase):
    def test_try_node_as_workflow_child(self):
        """TryNode can be used as a child node in a WorkflowNode."""
        try_node = _make_valid_try_node()
        workflow = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"try_block": try_node},
            input_edges={
                model.TargetHandle(node="try_block", port="inp"): model.InputSource(
                    port="x"
                ),
            },
            edges={},
            output_edges={
                model.OutputTarget(port="y"): model.SourceHandle(
                    node="try_block", port="out"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["try_block"], model.TryNode)

    def test_try_node_multiple_exception_cases_as_workflow_child(self):
        """TryNode with multiple exception cases can be a workflow child."""
        try_node = _make_valid_try_node(n_exception_cases=3)
        workflow = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"try_block": try_node},
            input_edges={
                model.TargetHandle(node="try_block", port="inp"): model.InputSource(
                    port="x"
                ),
            },
            edges={},
            output_edges={
                model.OutputTarget(port="y"): model.SourceHandle(
                    node="try_block", port="out"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["try_block"], model.TryNode)
        self.assertEqual(len(workflow.nodes["try_block"].exception_cases), 3)

    def test_workflow_with_try_node_roundtrip(self):
        """Workflow containing TryNode serializes and deserializes correctly."""
        try_node = _make_valid_try_node(n_exception_cases=2)
        original = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"try_block": try_node},
            input_edges={
                model.TargetHandle(node="try_block", port="inp"): model.InputSource(
                    port="x"
                ),
            },
            edges={},
            output_edges={
                model.OutputTarget(port="y"): model.SourceHandle(
                    node="try_block", port="out"
                ),
            },
        )

        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = model.WorkflowNode.model_validate(data)
                self.assertIsInstance(restored.nodes["try_block"], model.TryNode)
                self.assertEqual(len(restored.nodes["try_block"].exception_cases), 2)


class TestExceptionCaseSerialization(unittest.TestCase):
    def test_exception_case_roundtrip(self):
        """ExceptionCase JSON roundtrip."""
        original = model.ExceptionCase(
            exceptions=["builtins.ValueError", "builtins.TypeError"],
            body=model.LabeledNode(label="handler", node=_make_except_body()),
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = model.ExceptionCase.model_validate(data)
                self.assertEqual(len(restored.exceptions), 2)
                self.assertEqual(restored.body.label, "handler")


if __name__ == "__main__":
    unittest.main()
