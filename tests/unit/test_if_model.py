import unittest

import pydantic

from flowrep import model


class TestIfNodeBasicConstruction(unittest.TestCase):
    def _make_condition(self, inputs=None, outputs=None):
        return model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=inputs or ["x"],
            outputs=outputs or ["result"],
        )

    def _make_body(self, inputs=None, outputs=None):
        return model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=inputs or ["x"],
            outputs=outputs or ["y"],
        )

    def _make_case(self, inputs=None, outputs=None):
        return model.CaseModel(
            condition=self._make_condition(inputs=inputs),
            body=self._make_body(outputs=outputs),
        )

    def _make_valid_if_node(self, n_cases=1):
        cases = [self._make_case() for _ in range(n_cases)]
        else_case = self._make_body()

        input_edges = {
            model.TargetHandle(
                node=model.IfNode.condition_name(i), port="x"
            ): model.SourceHandle(node=None, port="inp")
            for i in range(n_cases)
        }

        expected_sources = [
            model.SourceHandle(node=model.IfNode.body_name(i), port="y")
            for i in range(n_cases)
        ] + [model.SourceHandle(node=model.IfNode.else_name(), port="y")]

        output_edges = {model.TargetHandle(node=None, port="out"): expected_sources}

        return model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=cases,
            else_case=else_case,
            input_edges=input_edges,
            output_edges_matrix=output_edges,
        )

    def test_valid_single_case(self):
        """IfNode with one case should validate."""
        node = self._make_valid_if_node(n_cases=1)
        self.assertEqual(node.type, model.RecipeElementType.IF)
        self.assertEqual(len(node.cases), 1)

    def test_valid_multiple_cases(self):
        """IfNode with multiple cases should validate."""
        node = self._make_valid_if_node(n_cases=3)
        self.assertEqual(len(node.cases), 3)

    def test_type_field_immutable(self):
        """IfNode type field should be frozen."""
        node = self._make_valid_if_node()
        with self.assertRaises(pydantic.ValidationError) as ctx:
            node.type = model.RecipeElementType.WORKFLOW
        self.assertIn("frozen", str(ctx.exception).lower())


class TestIfNodeCasesValidation(unittest.TestCase):
    def _make_condition(self):
        return model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=["result"],
        )

    def _make_body(self):
        return model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )

    def _make_case(self):
        return model.CaseModel(
            condition=self._make_condition(),
            body=self._make_body(),
        )

    def test_empty_cases_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[],
                else_case=self._make_body(),
                input_edges={},
                output_edges_matrix={
                    model.TargetHandle(node=None, port="out"): [
                        model.SourceHandle(node=model.IfNode.else_name(), port="y")
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
            edges={
                model.TargetHandle(node="inner", port="a"): model.SourceHandle(
                    node=None, port="x"
                ),
                model.TargetHandle(node=None, port="result"): model.SourceHandle(
                    node="inner", port="b"
                ),
            },
        )

        case = model.CaseModel(
            condition=workflow_condition,
            body=self._make_body(),
        )

        node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=[case],
            else_case=self._make_body(),
            input_edges={
                model.TargetHandle(
                    node=model.IfNode.condition_name(0), port="x"
                ): model.SourceHandle(node=None, port="inp")
            },
            output_edges_matrix={
                model.TargetHandle(node=None, port="out"): [
                    model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                    model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                ]
            },
        )
        self.assertIsInstance(node.cases[0].condition, model.WorkflowNode)


class TestIfNodeInputEdgesValidation(unittest.TestCase):
    def _make_condition(self):
        return model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=["result"],
        )

    def _make_body(self):
        return model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )

    def _make_case(self):
        return model.CaseModel(
            condition=self._make_condition(),
            body=self._make_body(),
        )

    def _make_output_edges(self, n_cases):
        sources = [
            model.SourceHandle(node=model.IfNode.body_name(i), port="y")
            for i in range(n_cases)
        ] + [model.SourceHandle(node=model.IfNode.else_name(), port="y")]
        return {model.TargetHandle(node=None, port="out"): sources}

    def test_input_edges_source_with_node_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case()],
                else_case=self._make_body(),
                input_edges={
                    model.TargetHandle(
                        node=model.IfNode.condition_name(0), port="x"
                    ): model.SourceHandle(
                        node="some_node", port="y"
                    )  # Should be None
                },
                output_edges_matrix=self._make_output_edges(1),
            )
        exc_str = str(ctx.exception)
        self.assertIn("node=None", exc_str)
        self.assertIn("some_node", exc_str)

    def test_input_edges_valid_source_none(self):
        node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=[self._make_case()],
            else_case=self._make_body(),
            input_edges={
                model.TargetHandle(
                    node=model.IfNode.condition_name(0), port="x"
                ): model.SourceHandle(node=None, port="inp")
            },
            output_edges_matrix=self._make_output_edges(1),
        )
        self.assertEqual(len(node.input_edges), 1)

    def test_input_edges_invalid_target_node(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case()],
                else_case=self._make_body(),
                input_edges={
                    model.TargetHandle(
                        node="invalid_name", port="x"
                    ): model.SourceHandle(node=None, port="inp")
                },
                output_edges_matrix=self._make_output_edges(1),
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid_name", exc_str)

    def test_input_edges_target_out_of_range(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case()],  # Only 1 case (index 0)
                else_case=self._make_body(),
                input_edges={
                    model.TargetHandle(
                        node=model.IfNode.condition_name(0), port="x"
                    ): model.SourceHandle(node=None, port="inp"),
                    model.TargetHandle(
                        node=model.IfNode.condition_name(5), port="x"
                    ): model.SourceHandle(  # Out of range
                        node=None, port="inp"
                    ),
                },
                output_edges_matrix=self._make_output_edges(1),
            )
        exc_str = str(ctx.exception)
        self.assertIn("condition_5", exc_str)

    def test_input_edges_can_target_bodies(self):
        """input_edges targets can include body nodes."""
        node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=[self._make_case()],  # body has input "x"
            else_case=self._make_body(),
            input_edges={
                model.TargetHandle(
                    node=model.IfNode.condition_name(0), port="x"
                ): model.SourceHandle(node=None, port="inp"),
                model.TargetHandle(
                    node=model.IfNode.body_name(0), port="x"
                ): model.SourceHandle(node=None, port="inp"),
            },
            output_edges_matrix=self._make_output_edges(1),
        )
        self.assertEqual(len(node.input_edges), 2)

    def test_input_edges_can_target_else(self):
        """input_edges targets can include the else node."""
        node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=[self._make_case()],
            else_case=self._make_body(),  # has input "x"
            input_edges={
                model.TargetHandle(
                    node=model.IfNode.condition_name(0), port="x"
                ): model.SourceHandle(node=None, port="inp"),
                model.TargetHandle(
                    node=model.IfNode.else_name(), port="x"
                ): model.SourceHandle(node=None, port="inp"),
            },
            output_edges_matrix=self._make_output_edges(1),
        )
        self.assertEqual(len(node.input_edges), 2)


class TestIfNodeOutputEdgesMatrixValidation(unittest.TestCase):
    def _make_condition(self):
        return model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=["result"],
        )

    def _make_body(self):
        return model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )

    def _make_case(self):
        return model.CaseModel(
            condition=self._make_condition(),
            body=self._make_body(),
        )

    def _make_input_edges(self, n_cases):
        return {
            model.TargetHandle(
                node=model.IfNode.condition_name(i), port="x"
            ): model.SourceHandle(node=None, port="inp")
            for i in range(n_cases)
        }

    def test_output_edges_matrix_target_with_node_rejected(self):
        """output_edges targets must have node=None."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case()],
                else_case=self._make_body(),
                input_edges=self._make_input_edges(1),
                output_edges_matrix={
                    model.TargetHandle(
                        node="some_node", port="out"
                    ): [  # Should be None
                        model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                        model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("node=None", exc_str)
        self.assertIn("some_node", exc_str)

    def test_output_edges_matrix_valid_target_none(self):
        node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=[self._make_case()],
            else_case=self._make_body(),
            input_edges=self._make_input_edges(1),
            output_edges_matrix={
                model.TargetHandle(node=None, port="out"): [
                    model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                    model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                ]
            },
        )
        self.assertEqual(len(node.output_edges_matrix), 1)

    def test_output_edges_matrix_missing_body_source(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case(), self._make_case()],
                else_case=self._make_body(),
                input_edges=self._make_input_edges(2),
                output_edges_matrix={
                    model.TargetHandle(node=None, port="out"): [
                        model.SourceHandle(
                            node=model.IfNode.body_name(0), port="y"
                        ),  # Missing body_1
                        model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid sources", exc_str)

    def test_output_edges_matrix_missing_else_source(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case()],
                else_case=self._make_body(),
                input_edges=self._make_input_edges(1),
                output_edges_matrix={
                    model.TargetHandle(node=None, port="out"): [
                        model.SourceHandle(
                            node=model.IfNode.body_name(0), port="y"
                        ),  # Missing else
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid sources", exc_str)

    def test_output_edges_matrix_extra_source_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case()],
                else_case=self._make_body(),
                input_edges=self._make_input_edges(1),
                output_edges_matrix={
                    model.TargetHandle(node=None, port="out"): [
                        model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                        model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                        model.SourceHandle(node="extra_node", port="z"),  # Extra source
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("invalid sources", exc_str)

    def test_output_edges_matrix_keys_must_match_outputs(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out", "other"],  # Two outputs declared
                cases=[self._make_case()],
                else_case=self._make_body(),
                input_edges=self._make_input_edges(1),
                output_edges_matrix={
                    model.TargetHandle(node=None, port="out"): [  # Only one edge
                        model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                        model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("must match outputs", exc_str)
        self.assertIn("other", exc_str)

    def test_output_edges_matrix_extra_key_rejected(self):
        """output_edges cannot have keys not in outputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],  # Only one output
                cases=[self._make_case()],
                else_case=self._make_body(),
                input_edges=self._make_input_edges(1),
                output_edges_matrix={
                    model.TargetHandle(node=None, port="out"): [
                        model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                        model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                    ],
                    model.TargetHandle(node=None, port="extra"): [  # Not in outputs
                        model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                        model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                    ],
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("must match outputs", exc_str)
        self.assertIn("extra", exc_str)

    def test_output_edges_matrix_match_case_count(self):
        body_node = model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y", "z"],
        )
        case = model.CaseModel(
            condition=model.AtomicNode(
                fully_qualified_name="mod.check",
                inputs=["x"],
                outputs=["result"],
            ),
            body=body_node,
        )
        node = model.IfNode(
            inputs=["inp"],
            outputs=["out1", "out2"],
            cases=[case],
            else_case=body_node,
            input_edges=self._make_input_edges(1),
            output_edges_matrix={
                model.TargetHandle(node=None, port="out1"): [
                    model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                    model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                ],
                model.TargetHandle(node=None, port="out2"): [
                    model.SourceHandle(node=model.IfNode.body_name(0), port="z"),
                    model.SourceHandle(node=model.IfNode.else_name(), port="z"),
                ],
            },
        )
        self.assertEqual(len(node.output_edges_matrix), len(node.cases) + 1)


class TestIfNodeNameConventions(unittest.TestCase):
    def test_condition_name_format(self):
        """condition_name should return 'condition_N'."""
        self.assertEqual(model.IfNode.condition_name(0), "condition_0")
        self.assertEqual(model.IfNode.condition_name(5), "condition_5")
        self.assertEqual(model.IfNode.condition_name(99), "condition_99")

    def test_body_name_format(self):
        """body_name should return 'body_N'."""
        self.assertEqual(model.IfNode.body_name(0), "body_0")
        self.assertEqual(model.IfNode.body_name(5), "body_5")
        self.assertEqual(model.IfNode.body_name(99), "body_99")

    def test_else_name_format(self):
        """else_name should return 'else'."""
        self.assertEqual(model.IfNode.else_name(), "else")


class TestIfNodeSerialization(unittest.TestCase):
    def _make_valid_if_node(self):
        condition = model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=["result"],
        )
        body = model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )
        case = model.CaseModel(condition=condition, body=body)
        return model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=[case],
            else_case=body,
            input_edges={
                model.TargetHandle(
                    node=model.IfNode.condition_name(0), port="x"
                ): model.SourceHandle(node=None, port="inp")
            },
            output_edges_matrix={
                model.TargetHandle(node=None, port="out"): [
                    model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                    model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                ]
            },
        )

    def test_json_roundtrip(self):
        original = self._make_valid_if_node()
        data = original.model_dump(mode="json")
        restored = model.IfNode.model_validate(data)

        self.assertEqual(original.inputs, restored.inputs)
        self.assertEqual(original.outputs, restored.outputs)
        self.assertEqual(len(original.cases), len(restored.cases))
        self.assertEqual(original.type, restored.type)

    def test_python_roundtrip(self):
        original = self._make_valid_if_node()
        data = original.model_dump(mode="python")
        restored = model.IfNode.model_validate(data)

        self.assertEqual(original.inputs, restored.inputs)
        self.assertEqual(original.outputs, restored.outputs)
        self.assertEqual(len(original.cases), len(restored.cases))

    def test_discriminated_union_roundtrip(self):
        original = self._make_valid_if_node()
        data = original.model_dump(mode="json")

        node = pydantic.TypeAdapter(model.NodeType).validate_python(data)
        self.assertIsInstance(node, model.IfNode)


class TestIfNodeInWorkflow(unittest.TestCase):
    def test_if_node_as_workflow_child(self):
        condition = model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=["result"],
        )
        body = model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )
        case = model.CaseModel(condition=condition, body=body)
        if_node = model.IfNode(
            inputs=["inp"],
            outputs=["out"],
            cases=[case],
            else_case=body,
            input_edges={
                model.TargetHandle(
                    node=model.IfNode.condition_name(0), port="x"
                ): model.SourceHandle(node=None, port="inp")
            },
            output_edges_matrix={
                model.TargetHandle(node=None, port="out"): [
                    model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                    model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                ]
            },
        )

        workflow = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"if_block": if_node},
            edges={
                model.TargetHandle(node="if_block", port="inp"): model.SourceHandle(
                    node=None, port="x"
                ),
                model.TargetHandle(node=None, port="y"): model.SourceHandle(
                    node="if_block", port="out"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["if_block"], model.IfNode)
        self.assertEqual(len(workflow.edges), 2)


class TestIfNodeInputEdgesPortValidation(unittest.TestCase):
    def _make_condition(self):
        return model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=["result"],
        )

    def _make_body(self):
        return model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )

    def _make_case(self):
        return model.CaseModel(
            condition=self._make_condition(),
            body=self._make_body(),
        )

    def _make_output_edges(self, n_cases):
        sources = [
            model.SourceHandle(node=model.IfNode.body_name(i), port="y")
            for i in range(n_cases)
        ] + [model.SourceHandle(node=model.IfNode.else_name(), port="y")]
        return {model.TargetHandle(node=None, port="out"): sources}

    def test_input_edges_invalid_target_port(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case()],  # condition has input "x"
                else_case=self._make_body(),
                input_edges={
                    model.TargetHandle(
                        node=model.IfNode.condition_name(0), port="nonexistent"
                    ): model.SourceHandle(node=None, port="inp")
                },
                output_edges_matrix=self._make_output_edges(1),
            )
        exc_str = str(ctx.exception)
        self.assertIn("has no input port", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_input_edges_valid_target_port(self):
        condition = model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["a", "b"],
            outputs=["result"],
        )
        case = model.CaseModel(condition=condition, body=self._make_body())
        node = model.IfNode(
            inputs=["inp1", "inp2"],
            outputs=["out"],
            cases=[case],
            else_case=self._make_body(),
            input_edges={
                model.TargetHandle(
                    node=model.IfNode.condition_name(0), port="a"
                ): model.SourceHandle(node=None, port="inp1"),
                model.TargetHandle(
                    node=model.IfNode.condition_name(0), port="b"
                ): model.SourceHandle(node=None, port="inp2"),
            },
            output_edges_matrix={
                model.TargetHandle(node=None, port="out"): [
                    model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                    model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                ]
            },
        )
        self.assertEqual(len(node.input_edges), 2)


class TestIfNodeOutputEdgesMatrixPortValidation(unittest.TestCase):
    def _make_condition(self):
        return model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=["result"],
        )

    def _make_body(self):
        return model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )

    def _make_case(self):
        return model.CaseModel(
            condition=self._make_condition(),
            body=self._make_body(),
        )

    def _make_input_edges(self, n_cases):
        return {
            model.TargetHandle(
                node=model.IfNode.condition_name(i), port="x"
            ): model.SourceHandle(node=None, port="inp")
            for i in range(n_cases)
        }

    def test_output_edges_matrix_invalid_body_source_port(self):
        """output_edges source port must exist on the body node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case()],  # body has output "y"
                else_case=self._make_body(),
                input_edges=self._make_input_edges(1),
                output_edges_matrix={
                    model.TargetHandle(node=None, port="out"): [
                        model.SourceHandle(
                            node=model.IfNode.body_name(0), port="nonexistent"
                        ),
                        model.SourceHandle(node=model.IfNode.else_name(), port="y"),
                    ]
                },
            )
        exc_str = str(ctx.exception)
        self.assertIn("has no output port", exc_str)
        self.assertIn("nonexistent", exc_str)

    def test_output_edges_matrix_invalid_else_source_port(self):
        """output_edges source port must exist on the else node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.IfNode(
                inputs=["inp"],
                outputs=["out"],
                cases=[self._make_case()],
                else_case=self._make_body(),  # has output "y"
                input_edges=self._make_input_edges(1),
                output_edges_matrix={
                    model.TargetHandle(node=None, port="out"): [
                        model.SourceHandle(node=model.IfNode.body_name(0), port="y"),
                        model.SourceHandle(
                            node=model.IfNode.else_name(), port="nonexistent"
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
        case = model.CaseModel(condition=self._make_condition(), body=body_node)
        node = model.IfNode(
            inputs=["inp"],
            outputs=["a", "b"],
            cases=[case],
            else_case=body_node,
            input_edges=self._make_input_edges(1),
            output_edges_matrix={
                model.TargetHandle(node=None, port="a"): [
                    model.SourceHandle(node=model.IfNode.body_name(0), port="out1"),
                    model.SourceHandle(node=model.IfNode.else_name(), port="out1"),
                ],
                model.TargetHandle(node=None, port="b"): [
                    model.SourceHandle(node=model.IfNode.body_name(0), port="out2"),
                    model.SourceHandle(node=model.IfNode.else_name(), port="out2"),
                ],
            },
        )
        self.assertEqual(len(node.output_edges_matrix), 2)


class TestCaseModelValidation(unittest.TestCase):
    def _make_condition(self, outputs=None):
        return model.AtomicNode(
            fully_qualified_name="mod.check",
            inputs=["x"],
            outputs=outputs or ["result"],
        )

    def _make_body(self):
        return model.AtomicNode(
            fully_qualified_name="mod.handle",
            inputs=["x"],
            outputs=["y"],
        )

    def test_single_output_condition_no_explicit_output(self):
        """CaseModel with single-output condition needs no condition_output."""
        case = model.CaseModel(
            condition=self._make_condition(outputs=["result"]),
            body=self._make_body(),
        )
        self.assertIsNone(case.condition_output)

    def test_multi_output_condition_requires_explicit_output(self):
        """CaseModel with multi-output condition must specify condition_output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.CaseModel(
                condition=self._make_condition(outputs=["a", "b"]),
                body=self._make_body(),
            )
        self.assertIn("exactly one output", str(ctx.exception))

    def test_multi_output_condition_with_valid_output(self):
        """CaseModel with explicit valid condition_output should pass."""
        case = model.CaseModel(
            condition=self._make_condition(outputs=["a", "b"]),
            body=self._make_body(),
            condition_output="a",
        )
        self.assertEqual(case.condition_output, "a")

    def test_invalid_condition_output_rejected(self):
        """CaseModel with invalid condition_output should fail."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.CaseModel(
                condition=self._make_condition(outputs=["a", "b"]),
                body=self._make_body(),
                condition_output="nonexistent",
            )
        self.assertIn("nonexistent", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
