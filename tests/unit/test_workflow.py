import unittest

import networkx as nx
from semantikon import datastructure
from semantikon.converter import parse_input_args

import flowrep.workflow as fwf


def operation(x: float, y: float) -> tuple[float, float]:
    return x + y, x - y


def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


def multiply(x: float, y: float = 5) -> float:
    return x * y


@fwf.workflow
def example_macro(a=10, b=20):
    c, d = operation(a, b)
    e = add(c, y=d)
    f = multiply(e)
    return f


@fwf.workflow
def example_workflow(a=10, b=20):
    y = example_macro(a, b)
    z = add(y, b)
    return z


@fwf.workflow
def parallel_execution(a=10, b=20):
    c = add(a)
    d = multiply(b)
    e, f = operation(c, d)
    return e, f


def my_while_condition(a=10, b=20):
    return a < b


def workflow_with_while(a=10, b=20):
    x = add(a, b)
    while my_while_condition(x, b):
        x = add(a, b)
        # Poor implementation to define variable inside the loop, but allowed
        z = multiply(a, x)
    return z


def complex_function(x=2.0, y=1) -> tuple[float, float]:
    speed = x / y
    return x, speed, speed / y


@fwf.workflow
def complex_macro(x=2.0):
    a, b, c = complex_function(x)
    return b, c


@fwf.workflow
def complex_workflow(x=2.0):
    b, c = complex_macro(x)
    return c


def seemingly_cyclic_workflow(a=10, b=20):
    a = add(a, b)
    return a


def workflow_to_use_undefined_variable(a=10, b=20):
    result = add(a, u)
    return result


def reused_args(a=10, b=20):
    a, b = operation(a, b)
    f = add(a, y=b)
    f = multiply(f)
    return f


def check_positive(x):
    if x < 0:
        raise ValueError("It must not be negative")


def workflow_with_leaf(x):
    y = add(x, x)
    check_positive(y)
    return y


def my_condition(X, Y):
    return X + Y < 100


def my_for_loop(a=10, b=20):
    return range(int(a), int(b))


def multiple_nested_workflow(a=1, b=2, c=3):
    d = add(a, b)
    e = multiply(b, c)
    f = add(c, d)

    while my_condition(d, e):
        d = add(d, b)
        e = multiply(e, c)

        for ii in my_for_loop(a, d):
            a = add(a, ii)
            d = multiply(b, e)

            while my_condition(c, f):
                c = add(c, a)
                f = multiply(f, b)

        while my_condition(b, f):
            b = add(b, b)
            f = multiply(a, b)

        while my_condition(c, e):
            c = add(c, d)
            e = multiply(e, c)

    while my_condition(a, b):
        a = add(a, c)
        b = multiply(b, d)

        while my_condition(d, f):
            d = add(d, b)
            f = multiply(f, a)

        while my_condition(e, f):
            e = add(e, a)
            f = multiply(f, c)

    while my_condition(f, f):
        f = add(f, f)
        f = multiply(f, f)

    return f


def workflow_with_for(a=10, b=20):
    x = add(a, b)
    for ii in my_for_loop(x, b):
        x = add(a, ii)
        z = multiply(a, x)
    return z


def my_if_condition(a=10, b=20):
    return a > b


def workflow_with_if(a=10, b=20):
    x = add(a, b)
    if my_if_condition(x, b):
        x = multiply(x, b)
    return x


def workflow_with_if_else(a=10, b=20):
    x = add(a, b)
    if my_if_condition(x, b):
        x = multiply(x, b)
    else:
        x = multiply(x, a)
        x = multiply(x, a)
    return x


class TestWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None

    def test_analyzer(self):
        graph = fwf.analyze_function(example_macro)[0]
        all_data = [
            ("operation_0", "c_0", {"type": "output", "output_index": 0}),
            ("operation_0", "d_0", {"type": "output", "output_index": 1}),
            ("c_0", "add_0", {"type": "input", "input_index": 0}),
            ("d_0", "add_0", {"type": "input", "input_name": "y"}),
            ("a_0", "operation_0", {"type": "input", "input_index": 0}),
            ("b_0", "operation_0", {"type": "input", "input_index": 1}),
            ("add_0", "e_0", {"type": "output"}),
            ("e_0", "multiply_0", {"type": "input", "input_index": 0}),
            ("multiply_0", "f_0", {"type": "output"}),
            ("f_0", "output", {"type": "input"}),
            ("input", "a_0", {"type": "output"}),
            ("input", "b_0", {"type": "output"}),
        ]
        self.maxDiff = None
        self.assertEqual(
            sorted([data for data in graph.edges.data()]),
            sorted(all_data),
        )

    def test_get_node_dict(self):
        node_dict = fwf.get_node_dict(add)
        self.assertEqual(
            node_dict,
            {
                "function": add,
                "type": "Function",
            },
        )

    def test_get_output_counts(self):
        graph = fwf.analyze_function(example_macro)[0]
        output_counts = fwf._get_output_counts(graph)
        self.assertEqual(output_counts, {"operation_0": 2, "add_0": 1, "multiply_0": 1})

    def test_get_workflow_dict(self):
        ref_data = {
            "inputs": {"a": {"default": 10}, "b": {"default": 20}},
            "outputs": {"f": {}},
            "nodes": {
                "operation_0": {
                    "function": {
                        "module": operation.__module__,
                        "qualname": operation.__qualname__,
                        "version": "not_defined",
                    },
                    "type": "Function",
                },
                "add_0": {
                    "function": {
                        "module": add.__module__,
                        "qualname": add.__qualname__,
                        "version": "not_defined",
                    },
                    "type": "Function",
                },
                "multiply_0": {
                    "function": {
                        "module": multiply.__module__,
                        "qualname": multiply.__qualname__,
                        "version": "not_defined",
                    },
                    "type": "Function",
                },
            },
            "edges": [
                ("inputs.a", "operation_0.inputs.0"),
                ("inputs.b", "operation_0.inputs.1"),
                ("operation_0.outputs.0", "add_0.inputs.0"),
                ("operation_0.outputs.1", "add_0.inputs.y"),
                ("add_0.outputs.output", "multiply_0.inputs.0"),
                ("multiply_0.outputs.output", "outputs.f"),
            ],
            "label": "example_macro",
            "type": "Workflow",
        }
        self.assertEqual(
            fwf.serialize_functions(example_macro._semantikon_workflow), ref_data
        )

    def test_get_workflow_dict_macro(self):
        result = fwf.get_workflow_dict(example_workflow)
        ref_data = {
            "inputs": {"a": {"default": 10}, "b": {"default": 20}},
            "outputs": {"z": {}},
            "nodes": {
                "example_macro_0": {
                    "inputs": {"a": {"default": 10}, "b": {"default": 20}},
                    "outputs": {"f": {}},
                    "nodes": {
                        "operation_0": {
                            "function": {
                                "module": operation.__module__,
                                "qualname": operation.__qualname__,
                                "version": "not_defined",
                            },
                            "type": "Function",
                        },
                        "add_0": {
                            "function": {
                                "module": add.__module__,
                                "qualname": add.__qualname__,
                                "version": "not_defined",
                            },
                            "type": "Function",
                        },
                        "multiply_0": {
                            "function": {
                                "module": multiply.__module__,
                                "qualname": multiply.__qualname__,
                                "version": "not_defined",
                            },
                            "type": "Function",
                        },
                    },
                    "edges": [
                        ("inputs.a", "operation_0.inputs.0"),
                        ("inputs.b", "operation_0.inputs.1"),
                        ("operation_0.outputs.0", "add_0.inputs.0"),
                        ("operation_0.outputs.1", "add_0.inputs.y"),
                        ("add_0.outputs.output", "multiply_0.inputs.0"),
                        ("multiply_0.outputs.output", "outputs.f"),
                    ],
                    "label": "example_macro_0",
                    "type": "Workflow",
                },
                "add_0": {
                    "function": {
                        "module": add.__module__,
                        "qualname": add.__qualname__,
                        "version": "not_defined",
                    },
                    "type": "Function",
                },
            },
            "edges": [
                ("inputs.a", "example_macro_0.inputs.0"),
                ("inputs.b", "example_macro_0.inputs.1"),
                ("inputs.b", "add_0.inputs.1"),
                ("example_macro_0.outputs.output", "add_0.inputs.0"),
                ("add_0.outputs.output", "outputs.z"),
            ],
            "label": "example_workflow",
            "type": "Workflow",
        }
        self.assertEqual(fwf.serialize_functions(result), ref_data, msg=result)

    def test_parallel_execution(self):
        graph = fwf.analyze_function(parallel_execution)[0]
        self.assertEqual(
            fwf.find_parallel_execution_levels(graph),
            [
                ["a_0", "b_0"],
                ["add_0", "multiply_0"],
                ["c_0", "d_0"],
                ["operation_0"],
                ["e_0", "f_0"],
            ],
        )

    def test_run_single(self):
        data = example_macro.run()
        self.assertEqual(example_macro(), data["outputs"]["f"]["value"])

    def test_run_parallel_execution(self):
        data = parallel_execution.run()
        results = parallel_execution()
        self.assertEqual(results[0], data["outputs"]["e"]["value"])
        self.assertEqual(results[1], data["outputs"]["f"]["value"])

    def test_run_nested(self):
        data = example_workflow.run()
        self.assertEqual(example_workflow(), data["outputs"]["z"]["value"])

    def test_not_implemented_error(self):
        def example_invalid_operator(a=10, b=20):
            y = example_macro(a, b)
            z = add(y, b)
            result = z + 1
            return result

        def example_invalid_multiple_operation(a=10, b=20):
            result = add(a, add(a, b))
            return result

        def example_invalid_local_var_def(a=10, b=20):
            result = add(a, 2)
            return result

        with self.assertRaises(NotImplementedError):
            fwf.workflow(example_invalid_operator)
        with self.assertRaises(NotImplementedError):
            fwf.workflow(example_invalid_multiple_operation)
        with self.assertRaises(NotImplementedError):
            fwf.workflow(example_invalid_local_var_def)

    def test_seemingly_cyclic_workflow(self):
        data = fwf.get_workflow_dict(seemingly_cyclic_workflow)
        self.assertIn("a", data["inputs"])
        self.assertIn("a", data["outputs"])

    def test_workflow_to_use_undefined_variable(self):
        with self.assertRaises(KeyError):
            fwf.workflow(workflow_to_use_undefined_variable)

    def test_get_sorted_edges(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")])
        sorted_edges = fwf._get_sorted_edges(graph)
        self.assertEqual(
            sorted_edges,
            [("A", "B", {}), ("A", "C", {}), ("B", "D", {}), ("C", "D", {})],
        )

    def test_workflow_with_while(self):
        wf = fwf.workflow(workflow_with_while)._semantikon_workflow
        self.assertIn("injected_While_0", wf["nodes"])
        self.assertEqual(
            sorted(wf["nodes"]["injected_While_0"]["inputs"].keys()),
            ["a", "b", "x"],
        )
        self.assertEqual(
            sorted(wf["nodes"]["injected_While_0"]["outputs"].keys()),
            ["x", "z"],
        )
        self.assertEqual(
            sorted(wf["nodes"]["injected_While_0"]["edges"]),
            sorted(
                [
                    ("inputs.x", "test.inputs.0"),
                    ("inputs.b", "test.inputs.1"),
                    ("inputs.b", "add_1.inputs.1"),
                    ("inputs.a", "add_1.inputs.0"),
                    ("inputs.a", "multiply_0.inputs.0"),
                    ("add_1.outputs.output", "multiply_0.inputs.1"),
                    ("multiply_0.outputs.output", "outputs.z"),
                    ("add_1.outputs.output", "outputs.x"),
                ]
            ),
        )
        self.assertIn("add_1", wf["nodes"]["injected_While_0"]["nodes"])
        self.assertIn("multiply_0", wf["nodes"]["injected_While_0"]["nodes"])

    def test_reused_args(self):
        data = fwf.get_workflow_dict(reused_args)
        self.assertEqual(
            sorted(data["edges"]),
            sorted(example_macro._semantikon_workflow["edges"]),
        )

    def test_get_node_outputs(self):
        self.assertEqual(
            fwf._get_node_outputs(operation, counts=2),
            {"output_0": {"dtype": float}, "output_1": {"dtype": float}},
        )
        self.assertEqual(
            fwf._get_node_outputs(operation, counts=1),
            {"output": {"dtype": tuple[float, float]}},
        )
        self.assertEqual(
            fwf._get_node_outputs(parallel_execution, counts=2),
            {"e": {}, "f": {}},
        )
        self.assertEqual(
            fwf._get_node_outputs(parallel_execution, counts=1),
            {"output": {}},
        )

    def test_workflow_with_leaf(self):
        data = fwf.get_workflow_dict(workflow_with_leaf)
        self.assertIn("check_positive_0", data["nodes"])
        self.assertIn("add_0", data["nodes"])
        self.assertIn("y", data["outputs"])
        self.assertIn(
            ("add_0.outputs.output", "check_positive_0.inputs.0"), data["edges"]
        )
        self.assertNotIn("outputs", data["nodes"]["check_positive_0"])

    def test_get_workflow_output(self):

        def test_function_1(a, b):
            return a + b

        self.assertEqual(
            fwf._get_node_outputs(test_function_1),
            {"output": {}},
        )

        def test_function_2(a, b):
            return a

        self.assertEqual(
            fwf._get_node_outputs(test_function_2),
            {"a": {}},
        )

        def test_function_3(a, b):
            return a, b

        self.assertEqual(
            fwf._get_node_outputs(test_function_3),
            {"a": {}, "b": {}},
        )

        def test_function_4(a, b):
            return a + b, b

        data = fwf._get_node_outputs(test_function_4)
        self.assertEqual(data, {"output_0": {}, "b": {}})
        data["output_0"]["value"] = 0
        self.assertEqual(
            data,
            {"output_0": {"value": 0}, "b": {}},
        )

        def test_function_5(a: int, b: int) -> tuple[int, int]:
            return a, b

        self.assertEqual(
            fwf._get_node_outputs(test_function_5),
            {"a": {"dtype": int}, "b": {"dtype": int}},
        )

    def test_detect_io_variables_from_control_flow(self):
        graph = fwf.analyze_function(workflow_with_while)[0]
        subgraphs = fwf._split_graphs_into_subgraphs(graph)
        io_vars = fwf._detect_io_variables_from_control_flow(
            graph, subgraphs["While_0"]
        )
        self.assertEqual(
            {key: sorted(value) for key, value in io_vars.items()},
            {
                "inputs": ["a_0", "b_0", "x_0"],
                "outputs": ["x_1", "z_0"],
            },
        )

    def test_get_control_flow_graph(self):
        control_flows = [
            "",
            "While_1/While_0",
            "While_2",
            "While_0",
            "While_1",
            "While_0/While_0/While_0",
            "While_1/While_1",
            "While_0/While_1",
            "While_0/While_0",
            "While_0/While_2",
        ]
        graph = fwf._get_control_flow_graph(control_flows)
        self.assertEqual(
            sorted(list(graph.successors("While_0"))),
            ["While_0/While_0", "While_0/While_1", "While_0/While_2"],
        )
        self.assertEqual(
            sorted(list(graph.successors("While_1"))),
            ["While_1/While_0", "While_1/While_1"],
        )

    def test_multiple_nested_workflow(self):
        data = fwf.get_workflow_dict(multiple_nested_workflow)
        self.assertIn("a", data["inputs"])
        self.assertIn("f", data["outputs"])
        self.assertIn("injected_While_0", data["nodes"])
        self.assertIn(
            "injected_While_0_While_0", data["nodes"]["injected_While_0"]["nodes"]
        )
        self.assertIn(
            "injected_While_0_For_0", data["nodes"]["injected_While_0"]["nodes"]
        )

    def test_for_loop(self):
        data = fwf.get_workflow_dict(workflow_with_for)
        self.assertIn("injected_For_0", data["nodes"])
        self.assertIn("iter", data["nodes"]["injected_For_0"])
        self.assertEqual(
            sorted(data["edges"]),
            sorted(
                [
                    ("inputs.a", "injected_For_0.inputs.a"),
                    ("inputs.b", "injected_For_0.inputs.b"),
                    ("inputs.a", "add_0.inputs.0"),
                    ("inputs.b", "add_0.inputs.1"),
                    ("add_0.outputs.output", "injected_For_0.inputs.x"),
                    ("injected_For_0.outputs.z", "outputs.z"),
                ]
            ),
        )
        self.assertEqual(
            sorted(data["nodes"]["injected_For_0"]["edges"]),
            sorted(
                [
                    ("inputs.x", "iter.inputs.0"),
                    ("inputs.b", "iter.inputs.1"),
                    ("iter.outputs.output", "add_1.inputs.1"),
                    ("inputs.a", "add_1.inputs.0"),
                    ("inputs.a", "multiply_0.inputs.0"),
                    ("add_1.outputs.output", "multiply_0.inputs.1"),
                    ("multiply_0.outputs.output", "outputs.z"),
                    ("add_1.outputs.output", "outputs.x"),
                ]
            ),
        )

    def test_multiple_output_to_single_raise_error(self):

        def multiple_output_to_single_variable(a, b):
            output = parallel_execution(a, b)
            return output

        self.assertRaises(
            ValueError, fwf.get_workflow_dict, multiple_output_to_single_variable
        )

    def test_if_statement(self):
        data = fwf.get_workflow_dict(workflow_with_if)
        self.assertIn("injected_If_0", data["nodes"])
        self.assertIn("x", data["outputs"])
        self.assertEqual(
            sorted(data["edges"]),
            sorted(
                [
                    ("inputs.b", "injected_If_0.inputs.b"),
                    ("inputs.a", "add_0.inputs.0"),
                    ("inputs.b", "add_0.inputs.1"),
                    ("add_0.outputs.output", "injected_If_0.inputs.x"),
                    ("injected_If_0.outputs.x", "outputs.x"),
                ]
            ),
        )
        self.assertEqual(
            sorted(data["nodes"]["injected_If_0"]["edges"]),
            sorted(
                [
                    ("inputs.x", "multiply_0.inputs.0"),
                    ("inputs.b", "multiply_0.inputs.1"),
                    ("inputs.x", "test.inputs.0"),
                    ("inputs.b", "test.inputs.1"),
                    ("multiply_0.outputs.output", "outputs.x"),
                ]
            ),
        )

    def test_if_else_statement(self):
        data = fwf.get_workflow_dict(workflow_with_if_else)
        self.assertIn("injected_If_0", data["nodes"])
        self.assertEqual(
            sorted(data["edges"]),
            sorted(
                [
                    ("inputs.b", "injected_If_0.inputs.b"),
                    ("inputs.a", "injected_Else_0.inputs.a"),
                    ("inputs.a", "add_0.inputs.0"),
                    ("inputs.b", "add_0.inputs.1"),
                    ("add_0.outputs.output", "injected_If_0.inputs.x"),
                    ("add_0.outputs.output", "injected_Else_0.inputs.x"),
                    ("injected_If_0.outputs.x", "outputs.x"),
                    ("injected_Else_0.outputs.x", "outputs.x"),
                ]
            ),
        )
        self.assertEqual(
            sorted(data["nodes"]["injected_If_0"]["edges"]),
            sorted(
                [
                    ("inputs.x", "multiply_0.inputs.0"),
                    ("inputs.b", "multiply_0.inputs.1"),
                    ("inputs.x", "test.inputs.0"),
                    ("inputs.b", "test.inputs.1"),
                    ("multiply_0.outputs.output", "outputs.x"),
                ]
            ),
        )
        self.assertEqual(
            sorted(data["nodes"]["injected_Else_0"]["edges"]),
            sorted(
                [
                    ("inputs.x", "multiply_1.inputs.0"),
                    ("inputs.x", "multiply_2.inputs.0"),  # This must not be here
                    ("inputs.a", "multiply_1.inputs.1"),
                    ("multiply_1.outputs.output", "multiply_2.inputs.0"),
                    ("inputs.a", "multiply_2.inputs.1"),
                    ("multiply_2.outputs.output", "outputs.x"),
                ]
            ),
        )

    def test_not_implemented_line_in_workflow(self):

        def f(x):
            def g(x):
                return x

            return g

        self.assertRaises(NotImplementedError, fwf.get_workflow_dict, f)

    def test_get_hashed_node_dict(self):

        def workflow_with_data(a=10, b=20):
            x = add(a, b)
            y = multiply(x, b)
            return x, y

        workflow_dict = fwf.get_workflow_dict(workflow_with_data)
        graph = fwf.get_workflow_graph(workflow_dict)
        data_dict = fwf.get_hashed_node_dict("add_0", graph, workflow_dict["nodes"])
        self.assertEqual(
            data_dict,
            {
                "node": {
                    "module": add.__module__,
                    "qualname": "add",
                    "version": "not_defined",
                    "connected_inputs": [],
                },
                "inputs": {"x": 10, "y": 20},
                "outputs": ["output"],
            },
        )
        add_hashed = fwf.get_node_hash("add_0", graph, workflow_dict["nodes"])
        data_dict = fwf.get_hashed_node_dict(
            "multiply_0", graph, workflow_dict["nodes"]
        )
        self.assertEqual(
            data_dict,
            {
                "node": {
                    "module": multiply.__module__,
                    "qualname": "multiply",
                    "version": "not_defined",
                    "connected_inputs": ["x"],
                },
                "inputs": {"x": add_hashed + "@output", "y": 20},
                "outputs": ["output"],
            },
        )
        graph = fwf.get_workflow_graph(example_workflow._semantikon_workflow)
        self.assertRaises(
            ValueError,
            fwf.get_hashed_node_dict,
            "add_0",
            graph,
            example_workflow._semantikon_workflow["nodes"],
        )

    def test_get_and_set_entry(self):

        def yet_another_workflow(a=10, b=20):
            x = add(a, b)
            y = multiply(x, b)
            return x, y

        workflow_dict = fwf.get_workflow_dict(yet_another_workflow)
        self.assertEqual(fwf._get_entry(workflow_dict, "inputs.a.default"), 10)
        self.assertRaises(KeyError, fwf._get_entry, workflow_dict, "inputs.x.default")
        fwf._set_entry(workflow_dict, "inputs.a.value", 42)
        self.assertEqual(fwf._get_entry(workflow_dict, "inputs.a.value"), 42)

    def test_get_function_metadata(self):
        self.assertEqual(
            fwf._get_function_metadata(operation),
            {
                "module": operation.__module__,
                "qualname": operation.__qualname__,
                "version": "not_defined",
            },
        )
        self.assertEqual(
            fwf._get_function_metadata(fwf._get_function_metadata(operation)),
            fwf._get_function_metadata(operation),
        )


if __name__ == "__main__":
    unittest.main()
