import unittest

import pydantic

from flowrep import model


def _make_condition(inputs=None, outputs=None) -> model.AtomicNode:
    return model.AtomicNode(
        fully_qualified_name="mod.check",
        inputs=inputs or ["x"],
        outputs=outputs or ["result"],
    )


def _make_body(inputs=None, outputs=None) -> model.AtomicNode:
    return model.AtomicNode(
        fully_qualified_name="mod.handle",
        inputs=inputs or ["x"],
        outputs=outputs or ["y"],
    )


def _make_case(n: int, inputs=None, outputs=None) -> model.ConditionalCase:
    return model.ConditionalCase(
        condition=model.LabeledNode(label=f"condition_{n}", node=_make_condition(inputs=inputs)),
        body=model.LabeledNode(label=f"body_{n}", node=_make_body(outputs=outputs)),
    )

def _make_else(inputs=None, outputs=None) -> model.LabeledNode:
    return model.LabeledNode(
        label="else_body", node=_make_body(inputs=inputs, outputs=outputs)
    )

def _make_input_edges(cases):
    return {
        model.TargetHandle(
            node=case.body.label, port="x"
        ): model.InputSource(port="inp")
        for case in cases
    }

def _make_output_edges(cases, else_case):
    sources = [
        model.SourceHandle(node=case.body.label, port="y")
        for case in cases
    ] + [model.SourceHandle(node=else_case.label, port="y")]
    return {model.OutputTarget(port="out"): sources}


def _make_valid_if_node(n_cases=1):
        cases = [_make_case(n) for n in range(n_cases)]
        else_case = _make_else()

        return model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,
            else_case=else_case,
            input_edges=_make_input_edges(cases),
            output_edges_matrix=_make_output_edges(cases, else_case)
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

    def test_type_field_immutable(self):
        """IfNode type field should be frozen."""
        node = _make_valid_if_node()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            node.type = model.RecipeElementType.WORKFLOW
        self.assertIn("frozen", str(ctx.exception).lower())


class TestIfNodeCasesValidation(unittest.TestCase):
    def test_empty_cases_rejected(self):
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[],
                else_case=else_case,
                input_edges={},
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node=else_case.label, port="y")
                    ]
                },
            )
        self.assertIn("at least one", str(ctx.exception))

    def test_cases_accepts_various_node_types(self):
        workflow_condition = model.WorkflowNode(
            inputs=["x"],
            outputs=["result"],
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
                model.OutputTarget(port="result"): model.SourceHandle(
                    node="inner", port="b"
                ),
            },
        )

        cases = [model.ConditionalCase(
            condition=model.LabeledNode(label="workflow_condition", node=workflow_condition),
            body=model.LabeledNode(label="body", node=_make_body()),
        )]
        else_case = _make_else()

        node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,
            else_case=else_case,
            input_edges=_make_input_edges(cases),
            output_edges_matrix=_make_output_edges(cases, else_case),
        )
        self.assertIsInstance(node.cases[0].condition.node, model.WorkflowNode)


class TestIfNodeInputEdgesValidation(unittest.TestCase):
    def test_input_edges_invalid_target_node(self):
        cases = [_make_case(0)]
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                else_case=else_case,
                input_edges={
                    model.TargetHandle(
                        node="invalid_name", port="x"
                    ): model.InputSource(port="inp")
                },
                output_edges_matrix=_make_output_edges(cases, else_case),
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid_name", exc_str)

    def test_input_edges_can_target_condition(self):
        """input_edges targets can include condition nodes."""
        cases = [_make_case(n) for n in range(2)]
        else_case = _make_else()
        node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,  # body has input "x"
            else_case=else_case,
            input_edges={
                model.TargetHandle(
                    node=cases[0].condition.label, port="x"
                ): model.InputSource(port="inp"),
                model.TargetHandle(
                    node=cases[1].condition.label, port="x"
                ): model.InputSource(port="inp"),
            },
            output_edges_matrix=_make_output_edges(cases, else_case),
        )
        self.assertEqual(len(node.input_edges), 2)

    def test_input_edges_can_target_bodies(self):
        """input_edges targets can include body nodes."""
        cases = [_make_case(n) for n in range(2)]
        else_case = _make_else()
        node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,  # body has input "x"
            else_case=else_case,
            input_edges={
                model.TargetHandle(
                    node=cases[0].body.label, port="x"
                ): model.InputSource(port="inp"),
                model.TargetHandle(
                    node=cases[1].body.label, port="x"
                ): model.InputSource(port="inp"),
            },
            output_edges_matrix=_make_output_edges(cases, else_case),
        )
        self.assertEqual(len(node.input_edges), 2)

    def test_input_edges_can_target_else(self):
        """input_edges targets can include the else node."""
        cases = [_make_case(n) for n in range(2)]
        else_case = _make_else()
        node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,  # body has input "x"
            else_case=else_case,
            input_edges={
                model.TargetHandle(
                    node=else_case.label, port="x"
                ): model.InputSource(port="inp"),
            },
            output_edges_matrix=_make_output_edges(cases, else_case),
        )
        self.assertEqual(len(node.input_edges), 1)


class TestIfNodeOutputEdgesMatrixValidation(unittest.TestCase):
    def test_output_edges_matrix_missing_body_source(self):
        cases = [_make_case(n) for n in range(2)]
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                else_case=else_case,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(
                            node=cases[0].body.label, port="y"
                        ),
                        # Missing cases[1].body.label entry
                        model.SourceHandle(node=else_case.label, port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid sources", exc_str)

    def test_output_edges_matrix_missing_else_source(self):
        cases = [_make_case(n) for n in range(2)]
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                else_case=else_case,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(
                            node=cases[0].body.label, port="y"
                        ),
                        model.SourceHandle(
                            node=cases[1].body.label, port="y"
                        ),
                        # Missing else
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid sources", exc_str)

    def test_output_edges_matrix_extra_source_rejected(self):
        cases = [_make_case(0) for n in range(2)]
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                else_case=else_case,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node=cases[0].body.label, port="y"),
                        model.SourceHandle(node=else_case.label, port="y"),
                        model.SourceHandle(node="extra_node", port="z"),  # Extra source
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid sources", exc_str)

    def test_output_edges_matrix_keys_must_match_outputs(self):
        cases = [_make_case(0) for n in range(2)]
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out", "other"],
                cases=cases,
                else_case=else_case,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node=cases[0].body.label, port="y"),
                        model.SourceHandle(node=else_case.label, port="y"),
                    ]
                    # Missing row of output mapping
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("must match outputs", exc_str)
        self.assertIn("other", exc_str)

    def test_output_edges_matrix_extra_key_rejected(self):
        """output_edges cannot have keys not in outputs."""
        cases = [_make_case(0) for n in range(2)]
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,
                else_case=else_case,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node=cases[0].body.label, port="y"),
                        model.SourceHandle(node=else_case.label, port="y"),
                    ],
                    model.OutputTarget(port="extra"): [  # Not in outputs
                        model.SourceHandle(node=cases[0].body.label, port="y"),
                        model.SourceHandle(node=else_case.label, port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("must match outputs", exc_str)
        self.assertIn("extra", exc_str)

    def test_output_edges_matrix_matches_outputs_and_case_shape(self):
        node = _make_valid_if_node()
        self.assertEqual(len(node.output_edges_matrix), len(node.outputs))
        for row in node.output_edges_matrix.values():
            with self.subTest(row=row):
                self.assertEqual(len(row), len(node.cases) + 1)


class TestIfNodeSerialization(unittest.TestCase):
    def test_roundtrip(self):
        original = _make_valid_if_node()
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = model.IfNode.model_validate(data)
                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(len(original.cases), len(restored.cases))
                self.assertEqual(original.type, restored.type)

    def test_roundtrip_multiple_cases(self):
        original = _make_valid_if_node(n_cases=3)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = model.IfNode.model_validate(data)
                self.assertEqual(len(restored.cases), 3)
                self.assertEqual(len(restored.input_edges), 3)
                self.assertEqual(len(restored.output_edges_matrix[model.OutputTarget(port="out")]), 4)

    def test_roundtrip_with_condition_output(self):
        condition = model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=["a", "b"],
        )
        body = model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )
        cases = [model.ConditionalCase(
            condition=model.LabeledNode(label="condition", node=condition),
            body=model.LabeledNode(label="body", node=body),
            condition_output="a"
        )]
        else_case = _make_else()
        original = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,
            else_case=else_case,
            input_edges=_make_input_edges(cases),
            output_edges_matrix=_make_output_edges(cases, else_case),
        )

        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = model.IfNode.model_validate(data)
                self.assertEqual(restored.cases[0].condition_output, "a")

    def test_discriminated_union_roundtrip(self):
        original = _make_valid_if_node()
        data = original.model_dump(mode="json")

        node = pydantic.TypeAdapter(model.NodeType).validate_python(data)
        self.assertIsInstance(node, model.IfNode)


class TestIfNodeInWorkflow(unittest.TestCase):
    def test_if_node_as_workflow_child(self):
        if_node = _make_valid_if_node()
        workflow = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"if_block": if_node},
            input_edges={
                model.TargetHandle(node="if_block", port="inp"): model.InputSource(
                    port="x"
                ),
            },
            edges={},
            output_edges={
                model.OutputTarget(port="y"): model.SourceHandle(
                    node="if_block", port="out"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["if_block"], model.IfNode)


class TestIfNodeInputEdgesPortValidation(unittest.TestCase):
    def test_input_edges_invalid_target_port(self):
        cases = [_make_case(0)]
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,  # condition has input "x"
                else_case=else_case,
                input_edges={
                    model.TargetHandle(
                        node=cases[0].body.label, port="nonexistent"
                    ): model.InputSource(port="inp")
                },
                output_edges_matrix=_make_output_edges(cases, else_case),
            )
        exc_str = str(ctx.exception)
        self.assertIn("has no input port", exc_str)
        self.assertIn("nonexistent", exc_str)


class TestIfNodeOutputEdgesMatrixPortValidation(unittest.TestCase):
    def test_output_edges_matrix_invalid_body_source_port(self):
        """output_edges source port must exist on the body node."""
        cases = [_make_case(0)]
        else_case = _make_else()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,  # body has output "y"
                else_case=else_case,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(
                            node=cases[0].body.label, port="nonexistent"
                        ),
                        model.SourceHandle(node=else_case.label, port="y"),
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
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=cases,  # body has output "y"
                else_case=else_case,
                input_edges=_make_input_edges(cases),
                output_edges_matrix={
                    model.OutputTarget(port="out"): [
                        model.SourceHandle(node=cases[0].body.label, port="y"),
                        model.SourceHandle(
                            node=else_case.label, port="nonexistent"
                        ),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("has no output port", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_output_edges_matrix_valid_source_ports(self):
        """output_edges with valid source ports should pass."""
        body_node = model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["out1", "out2"],
        )
        cases = [model.ConditionalCase(
            condition=model.LabeledNode(label="condition", node=_make_condition()),
            body=model.LabeledNode(label="body", node=body_node)
        )]
        node = model.IfNode(
            inputs=["inp"],
            outputs=["a", "b"],
            cases=cases,
            else_case=model.LabeledNode(label="else_case", node=body_node),
            input_edges=_make_input_edges(cases),
            output_edges_matrix={
                model.OutputTarget(port="a"): [
                    model.SourceHandle(node="body", port="out1"),
                    model.SourceHandle(node="else_case", port="out1"),
                ],
                model.OutputTarget(port="b"): [
                    model.SourceHandle(node="body", port="out2"),
                    model.SourceHandle(node="else_case", port="out2"),
                ],
            },
        )
        self.assertEqual(len(node.output_edges_matrix), 2)


class TestConditionalCaseValidation(unittest.TestCase):
    def _make_condition(self, outputs=None):
        return model.LabeledNode(
            label="condition",
            node=model.AtomicNode(
                fully_qualified_name="mod.check",
                inputs=["x"],
                outputs=outputs or ["result"],
            ),
        )

    def _make_body(self):
        return model.LabeledNode(
            label="body",
            node=model.AtomicNode(
                fully_qualified_name="mod.handle",
                inputs=["x"],
                outputs=["y"],
            ),
        )

    def test_single_output_condition_no_explicit_output(self):
        """ConditionalCase with single-output condition needs no condition_output."""
        case = model.ConditionalCase(
            condition=self._make_condition(outputs=["result"]),
            body=self._make_body(),
        )
        self.assertIsNone(case.condition_output)

    def test_multi_output_condition_requires_explicit_output(self):
        """ConditionalCase with multi-output condition must specify condition_output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ConditionalCase(
                condition=self._make_condition(outputs=["a", "b"]),
                body=self._make_body(),
            )
        self.assertIn("exactly one output", str(ctx.exception))

    def test_multi_output_condition_with_valid_output(self):
        """ConditionalCase with explicit valid condition_output should pass."""
        case = model.ConditionalCase(
            condition=self._make_condition(outputs=["a", "b"]),
            body=self._make_body(),
            condition_output="a",
        )
        self.assertEqual(case.condition_output, "a")

    def test_invalid_condition_output_rejected(self):
        """ConditionalCase with invalid condition_output should fail."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ConditionalCase(
                condition=self._make_condition(outputs=["a", "b"]),
                body=self._make_body(),
                condition_output="nonexistent",
            )
        self.assertIn("nonexistent", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
