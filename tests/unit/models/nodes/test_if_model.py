import unittest

import pydantic

from flowrep.models import edges
from flowrep.models.nodes import atomic_model, if_model, model, union, workflow_model


def _make_condition(inputs=None, outputs=None) -> atomic_model.AtomicNode:
    return atomic_model.AtomicNode(
        fully_qualified_name="mod.check",
        inputs=inputs or ["x"],
        outputs=outputs or ["result"],
    )


def _make_body(inputs=None, outputs=None) -> atomic_model.AtomicNode:
    return atomic_model.AtomicNode(
        fully_qualified_name="mod.handle",
        inputs=inputs or ["x"],
        outputs=outputs or ["y"],
    )


def _make_case(n: int, inputs=None, outputs=None) -> model.ConditionalCase:
    return model.ConditionalCase(
        condition=model.LabeledNode(
            label=f"condition_{n}", node=_make_condition(inputs=inputs)
        ),
        body=model.LabeledNode(label=f"body_{n}", node=_make_body(outputs=outputs)),
    )


def _make_else(inputs=None, outputs=None) -> model.LabeledNode:
    return model.LabeledNode(
        label="else_body", node=_make_body(inputs=inputs, outputs=outputs)
    )


def _make_input_edges(cases, else_case=None):
    edge_dict = {
        edges.TargetHandle(node=case.body.label, port="x"): edges.InputSource(
            port="inp"
        )
        for case in cases
    }
    if else_case is not None:
        edge_dict[edges.TargetHandle(node=else_case.label, port="x")] = (
            edges.InputSource(port="inp")
        )
    return edge_dict


def _make_output_edges(cases, else_case=None):
    sources = [edges.SourceHandle(node=case.body.label, port="y") for case in cases]
    if else_case is not None:
        sources.append(edges.SourceHandle(node=else_case.label, port="y"))
    return {edges.OutputTarget(port="out"): sources}


def _make_valid_if_node(n_cases=1, with_else=True):
    cases = [_make_case(n) for n in range(n_cases)]
    else_case = _make_else() if with_else else None

    return if_model.IfNode(
        inputs=["inp"],
        outputs=["out"],
        cases=cases,
        else_case=else_case,
        input_edges=_make_input_edges(cases, else_case),
        output_edges_matrix=_make_output_edges(cases, else_case),
    )


class TestIfNodeBasicConstruction(unittest.TestCase):
    def test_valid_single_case(self):
        """IfNode with one case should validate."""
        node = _make_valid_if_node(n_cases=1)
        self.assertEqual(node.type, model.RecipeElementType.IF)
        self.assertEqual(len(node.cases), 1)

    def test_valid_multiple_cases(self):
        """IfNode with multiple cases should validate."""
        node = _make_valid_if_node(n_cases=3)
        self.assertEqual(len(node.cases), 3)

    def test_valid_without_else_case(self):
        """IfNode without else_case should validate."""
        node = _make_valid_if_node(n_cases=2, with_else=False)
        self.assertIsNone(node.else_case)
        self.assertEqual(len(node.cases), 2)

    def test_type_field_immutable(self):
        """IfNode type field should be frozen."""
        node = _make_valid_if_node()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            node.type = model.RecipeElementType.WORKFLOW
        self.assertIn("frozen", str(ctx.exception).lower())


class TestIfNodeCasesValidation(unittest.TestCase):
    def test_empty_cases_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[],
                input_edges={},
                output_edges_matrix={edges.OutputTarget(port="out"): []},
            )
        self.assertIn("at least one", str(ctx.exception))

    def test_duplicate_labels_across_cases_rejected(self):
        """Labels must be unique across all conditions, bodies, and else_case."""
        case0 = model.ConditionalCase(
            condition=model.LabeledNode(label="cond_0", node=_make_condition()),
            body=model.LabeledNode(label="shared_label", node=_make_body()),
        )
        case1 = model.ConditionalCase(
            condition=model.LabeledNode(label="cond_1", node=_make_condition()),
            body=model.LabeledNode(
                label="shared_label", node=_make_body()
            ),  # Duplicate
        )
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[case0, case1],
                input_edges={},
                output_edges_matrix={
                    edges.OutputTarget(port="out"): [
                        edges.SourceHandle(node="shared_label", port="y"),
                        edges.SourceHandle(node="shared_label", port="y"),
                    ]
                },
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_cases_accepts_various_node_types(self):
        workflow_condition = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["result"],
            nodes={
                "inner": atomic_model.AtomicNode(
                    fully_qualified_name="mod.f",
                    inputs=["a"],
                    outputs=["b"],
                )
            },
            input_edges={
                edges.TargetHandle(node="inner", port="a"): edges.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edges.OutputTarget(port="result"): edges.SourceHandle(
                    node="inner", port="b"
                ),
            },
        )

        cases = [
            model.ConditionalCase(
                condition=model.LabeledNode(
                    label="workflow_condition", node=workflow_condition
                ),
                body=model.LabeledNode(label="body", node=_make_body()),
            )
        ]

        node = if_model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,
            input_edges=_make_input_edges(cases),
            output_edges_matrix=_make_output_edges(cases),
        )
        self.assertIsInstance(node.cases[0].condition.node, workflow_model.WorkflowNode)


class TestIfNodeInputEdgesValidation(unittest.TestCase):
    def test_input_edges_invalid_target_node(self):
        cases = [_make_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                input_edges={
                    edges.TargetHandle(
                        node="invalid_name", port="x"
                    ): edges.InputSource(port="inp")
                },
                output_edges_matrix=_make_output_edges(cases),
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid_name", exc_str)

    def test_input_edges_can_target_condition(self):
        """input_edges targets can include condition nodes."""
        cases = [_make_case(n) for n in range(2)]
        node = if_model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,  # body has input "x"
            input_edges={
                edges.TargetHandle(
                    node=cases[0].condition.label, port="x"
                ): edges.InputSource(port="inp"),
                edges.TargetHandle(
                    node=cases[1].condition.label, port="x"
                ): edges.InputSource(port="inp"),
            },
            output_edges_matrix=_make_output_edges(cases),
        )
        self.assertEqual(len(node.input_edges), 2)

    def test_input_edges_can_target_bodies(self):
        """input_edges targets can include body nodes."""
        cases = [_make_case(n) for n in range(2)]
        node = if_model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,  # body has input "x"
            input_edges={
                edges.TargetHandle(
                    node=cases[0].body.label, port="x"
                ): edges.InputSource(port="inp"),
                edges.TargetHandle(
                    node=cases[1].body.label, port="x"
                ): edges.InputSource(port="inp"),
            },
            output_edges_matrix=_make_output_edges(cases),
        )
        self.assertEqual(len(node.input_edges), 2)

    def test_input_edges_can_target_else(self):
        """input_edges targets can include the else node."""
        cases = [_make_case(n) for n in range(2)]
        else_case = _make_else()
        node = if_model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,  # body has input "x"
            else_case=else_case,
            input_edges={
                edges.TargetHandle(node=else_case.label, port="x"): edges.InputSource(
                    port="inp"
                ),
            },
            output_edges_matrix=_make_output_edges(cases, else_case),
        )
        self.assertEqual(len(node.input_edges), 1)


class TestIfNodeOutputEdgesMatrixValidation(unittest.TestCase):
    def test_output_edges_matrix_invalid_source_node(self):
        """Sources must reference valid prospective nodes."""
        cases = [_make_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    edges.OutputTarget(port="out"): [
                        edges.SourceHandle(node="nonexistent", port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid sources", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_output_edges_matrix_duplicate_source_node_rejected(self):
        """Each prospective node can appear at most once per output."""
        cases = [_make_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    edges.OutputTarget(port="out"): [
                        edges.SourceHandle(node=cases[0].body.label, port="y"),
                        edges.SourceHandle(node=cases[0].body.label, port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("at most one", exc_str)
        self.assertIn("duplicates", exc_str.lower())

    def test_output_edges_matrix_keys_must_match_outputs(self):
        cases = [_make_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out", "other"],
                cases=cases,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    edges.OutputTarget(port="out"): [
                        edges.SourceHandle(node=cases[0].body.label, port="y"),
                    ]
                    # Missing row for "other"
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("must match outputs", exc_str)
        self.assertIn("other", exc_str)

    def test_output_edges_matrix_extra_key_rejected(self):
        """output_edges cannot have keys not in outputs."""
        cases = [_make_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    edges.OutputTarget(port="out"): [
                        edges.SourceHandle(node=cases[0].body.label, port="y"),
                    ],
                    edges.OutputTarget(port="extra"): [
                        edges.SourceHandle(node=cases[0].body.label, port="y"),
                    ],
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("must match outputs", exc_str)
        self.assertIn("extra", exc_str)

    def test_output_edges_matrix_empty_sources_rejected(self):
        """An output must have at least one source."""
        cases = [_make_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={edges.OutputTarget(port="out"): []},
            )
        exc_str = str(ctx.exception)
        self.assertIn("at least one", exc_str)

    def test_output_edges_matrix_partial_sources_allowed(self):
        """An output can have sources from only some prospective nodes."""
        cases = [_make_case(n) for n in range(3)]
        else_case = _make_else()
        node = if_model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,
            else_case=else_case,
            input_edges=_make_input_edges(cases, else_case),
            output_edges_matrix={
                edges.OutputTarget(port="out"): [
                    # Only body_0 and else_body, skipping body_1 and body_2
                    edges.SourceHandle(node=cases[0].body.label, port="y"),
                    edges.SourceHandle(node=else_case.label, port="y"),
                ]
            },
        )
        self.assertEqual(
            len(node.output_edges_matrix[edges.OutputTarget(port="out")]), 2
        )

    def test_output_edges_matrix_all_sources_allowed(self):
        """An output can have sources from all prospective nodes."""
        cases = [_make_case(n) for n in range(2)]
        else_case = _make_else()
        node = if_model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,
            else_case=else_case,
            input_edges=_make_input_edges(cases, else_case),
            output_edges_matrix={
                edges.OutputTarget(port="out"): [
                    edges.SourceHandle(node=cases[0].body.label, port="y"),
                    edges.SourceHandle(node=cases[1].body.label, port="y"),
                    edges.SourceHandle(node=else_case.label, port="y"),
                ]
            },
        )
        self.assertEqual(
            len(node.output_edges_matrix[edges.OutputTarget(port="out")]), 3
        )

    def test_output_edges_matrix_can_source_from_conditions(self):
        """
        Sources can come from condition nodes, not just bodies.

        Subject to change -- I don't see a good reason now to disallow it, but it does
        feel a bit silly.
        """
        cases = [_make_case(0)]
        node = if_model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,
            input_edges=_make_input_edges(cases),
            output_edges_matrix={
                edges.OutputTarget(port="out"): [
                    edges.SourceHandle(node=cases[0].condition.label, port="result"),
                ]
            },
        )
        self.assertEqual(
            len(node.output_edges_matrix[edges.OutputTarget(port="out")]), 1
        )


class TestIfNodeProspectiveNodes(unittest.TestCase):
    def test_prospective_nodes_with_else(self):
        """prospective_nodes includes conditions, bodies, and else_case."""
        node = _make_valid_if_node(n_cases=2, with_else=True)
        prospective = node.prospective_nodes
        self.assertIn("condition_0", prospective)
        self.assertIn("condition_1", prospective)
        self.assertIn("body_0", prospective)
        self.assertIn("body_1", prospective)
        self.assertIn("else_body", prospective)
        self.assertEqual(len(prospective), 5)

    def test_prospective_nodes_without_else(self):
        """prospective_nodes excludes else_case when None."""
        node = _make_valid_if_node(n_cases=2, with_else=False)
        prospective = node.prospective_nodes
        self.assertIn("condition_0", prospective)
        self.assertIn("condition_1", prospective)
        self.assertIn("body_0", prospective)
        self.assertIn("body_1", prospective)
        self.assertNotIn("else_body", prospective)
        self.assertEqual(len(prospective), 4)


class TestIfNodeSerialization(unittest.TestCase):
    def test_roundtrip(self):
        original = _make_valid_if_node()
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = if_model.IfNode.model_validate(data)
                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(len(original.cases), len(restored.cases))
                self.assertEqual(original.type, restored.type)

    def test_roundtrip_without_else(self):
        original = _make_valid_if_node(n_cases=2, with_else=False)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = if_model.IfNode.model_validate(data)
                self.assertIsNone(restored.else_case)
                self.assertEqual(len(restored.cases), 2)

    def test_roundtrip_multiple_cases(self):
        original = _make_valid_if_node(n_cases=3)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = if_model.IfNode.model_validate(data)
                self.assertEqual(len(restored.cases), 3)
                self.assertEqual(len(restored.input_edges), 4)  # 3 bodies + 1 else
                self.assertEqual(
                    len(restored.output_edges_matrix[edges.OutputTarget(port="out")]), 4
                )

    def test_roundtrip_with_condition_output(self):
        condition = atomic_model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=["a", "b"],
        )
        body = atomic_model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )
        cases = [
            model.ConditionalCase(
                condition=model.LabeledNode(label="condition", node=condition),
                body=model.LabeledNode(label="body", node=body),
                condition_output="a",
            )
        ]
        original = if_model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,
            input_edges=_make_input_edges(cases),
            output_edges_matrix=_make_output_edges(cases),
        )

        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = if_model.IfNode.model_validate(data)
                self.assertEqual(restored.cases[0].condition_output, "a")

    def test_discriminated_union_roundtrip(self):
        original = _make_valid_if_node()
        data = original.model_dump(mode="json")

        node = pydantic.TypeAdapter(union.NodeType).validate_python(data)
        self.assertIsInstance(node, if_model.IfNode)

    def test_discriminated_union_roundtrip_without_else(self):
        original = _make_valid_if_node(with_else=False)
        data = original.model_dump(mode="json")

        node = pydantic.TypeAdapter(union.NodeType).validate_python(data)
        self.assertIsInstance(node, if_model.IfNode)
        self.assertIsNone(node.else_case)


class TestIfNodeInWorkflow(unittest.TestCase):
    def test_if_node_as_workflow_child(self):
        if_node = _make_valid_if_node()
        workflow = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"if_block": if_node},
            input_edges={
                edges.TargetHandle(node="if_block", port="inp"): edges.InputSource(
                    port="x"
                ),
            },
            edges={},
            output_edges={
                edges.OutputTarget(port="y"): edges.SourceHandle(
                    node="if_block", port="out"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["if_block"], if_model.IfNode)

    def test_if_node_without_else_as_workflow_child(self):
        if_node = _make_valid_if_node(with_else=False)
        workflow = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"if_block": if_node},
            input_edges={
                edges.TargetHandle(node="if_block", port="inp"): edges.InputSource(
                    port="x"
                ),
            },
            edges={},
            output_edges={
                edges.OutputTarget(port="y"): edges.SourceHandle(
                    node="if_block", port="out"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["if_block"], if_model.IfNode)
        self.assertIsNone(workflow.nodes["if_block"].else_case)


class TestIfNodeInputEdgesPortValidation(unittest.TestCase):
    def test_input_edges_invalid_target_port(self):
        cases = [_make_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,  # condition has input "x"
                input_edges={
                    edges.TargetHandle(
                        node=cases[0].body.label, port="nonexistent"
                    ): edges.InputSource(port="inp")
                },
                output_edges_matrix=_make_output_edges(cases),
            )
        exc_str = str(ctx.exception)
        self.assertIn("has no input port", exc_str)
        self.assertIn("nonexistent", exc_str)


class TestIfNodeOutputEdgesMatrixPortValidation(unittest.TestCase):
    def test_output_edges_matrix_invalid_body_source_port(self):
        """output_edges source port must exist on the body node."""
        cases = [_make_case(0)]
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,  # body has output "y"
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    edges.OutputTarget(port="out"): [
                        edges.SourceHandle(
                            node=cases[0].body.label, port="nonexistent"
                        ),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("has no output port", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_output_edges_matrix_invalid_else_source_port(self):
        """output_edges source port must exist on the else node."""
        cases = [_make_case(0)]
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            if_model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,  # body has output "y"
                else_case=else_case,
                input_edges=_make_input_edges(cases, else_case),
                output_edges_matrix={
                    edges.OutputTarget(port="out"): [
                        edges.SourceHandle(node=cases[0].body.label, port="y"),
                        edges.SourceHandle(node=else_case.label, port="nonexistent"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("has no output port", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_output_edges_matrix_valid_source_ports(self):
        """output_edges with valid source ports should pass."""
        body_node = atomic_model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["out1", "out2"],
        )
        cases = [
            model.ConditionalCase(
                condition=model.LabeledNode(label="condition", node=_make_condition()),
                body=model.LabeledNode(label="body", node=body_node),
            )
        ]
        node = if_model.IfNode(
            inputs=["inp"],
            outputs=["a", "b"],
            cases=cases,
            input_edges=_make_input_edges(cases),
            output_edges_matrix={
                edges.OutputTarget(port="a"): [
                    edges.SourceHandle(node="body", port="out1"),
                ],
                edges.OutputTarget(port="b"): [
                    edges.SourceHandle(node="body", port="out2"),
                ],
            },
        )
        self.assertEqual(len(node.output_edges_matrix), 2)

    def test_output_edges_matrix_valid_source_ports_with_else(self):
        """output_edges with valid source ports and else_case should pass."""
        body_node = atomic_model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["out1", "out2"],
        )
        cases = [
            model.ConditionalCase(
                condition=model.LabeledNode(label="condition", node=_make_condition()),
                body=model.LabeledNode(label="body", node=body_node),
            )
        ]
        node = if_model.IfNode(
            inputs=["inp"],
            outputs=["a", "b"],
            cases=cases,
            else_case=model.LabeledNode(label="else_case", node=body_node),
            input_edges=_make_input_edges(cases),
            output_edges_matrix={
                edges.OutputTarget(port="a"): [
                    edges.SourceHandle(node="body", port="out1"),
                    edges.SourceHandle(node="else_case", port="out1"),
                ],
                edges.OutputTarget(port="b"): [
                    edges.SourceHandle(node="body", port="out2"),
                    edges.SourceHandle(node="else_case", port="out2"),
                ],
            },
        )
        self.assertEqual(len(node.output_edges_matrix), 2)


if __name__ == "__main__":
    unittest.main()
