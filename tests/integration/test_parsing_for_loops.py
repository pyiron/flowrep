import unittest

from flowrep.models.nodes import atomic_model, for_model, workflow_model
from flowrep.models.parsers import workflow_parser


def my_unity(x):
    return x


def my_range(n: int) -> list[int]:
    return list(range(n))


def how_many(lst: list) -> int:
    length = len(lst)
    return length


def single_loop(ns):
    """
    Here, we achieve the same result by decomposing into two legitimate loops
    """
    vecs = []

    pass_through = my_unity(ns)

    for n in pass_through:
        x = my_unity(n)
        rs = my_range(x)
        vecs.append(rs)

    l = how_many(vecs)  # noqa: E741
    return l, vecs


unity_node = atomic_model.AtomicNode.model_validate(
    {
        "type": "atomic",
        "fully_qualified_name": f"{my_unity.__module__}.{my_unity.__qualname__}",
        "inputs": ["x"],
        "outputs": ["x"],
    }
)

range_node = atomic_model.AtomicNode.model_validate(
    {
        "type": "atomic",
        "fully_qualified_name": f"{my_range.__module__}.{my_range.__qualname__}",
        "inputs": ["n"],
        "outputs": ["output_0"],
    }
)

for_body = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["n"],
        "outputs": ["rs"],
        "nodes": {
            "my_unity_0": unity_node,
            "my_range_0": range_node,
        },
        "input_edges": {"my_unity_0.x": "n"},
        "edges": {"my_range_0.n": "my_unity_0.x"},
        "output_edges": {"rs": "my_range_0.output_0"},
    }
)

for_node = for_model.ForNode.model_validate(
    {
        "type": "for",
        "inputs": ["pass_through"],
        "outputs": ["vecs"],
        "body_node": {"label": "body", "node": for_body},
        "input_edges": {"body.n": "pass_through"},
        "output_edges": {"vecs": "body.rs"},
        "nested_ports": ["n"],
        "zipped_ports": [],
        "transfer_edges": {},
    }
)

how_node = atomic_model.AtomicNode.model_validate(
    {
        "type": "atomic",
        "fully_qualified_name": f"{how_many.__module__}.{how_many.__qualname__}",
        "inputs": ["lst"],
        "outputs": ["length"],
    }
)

single_loop_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["ns"],
        "outputs": ["l", "vecs"],
        "nodes": {
            "my_unity_0": unity_node,
            "for_0": for_node,
            "how_many_0": how_node,
        },
        "input_edges": {"my_unity_0.x": "ns"},
        "edges": {
            "for_0.pass_through": "my_unity_0.x",
            "how_many_0.lst": "for_0.vecs",
        },
        "output_edges": {
            "l": "how_many_0.length",
            "vecs": "for_0.vecs",
        },
    }
)


def takes_many(a, b, c, d):
    total = a + b + c + d
    return total


def zipped_broadcast_and_transferred(a, bs, cs, ds):
    b_accumulator = []
    c_accumulator = []
    d_accumulator = []
    sums = []
    for b in bs:
        for c, d in zip(cs, ds, strict=True):
            b_accumulator.append(b)
            c_accumulator.append(c)
            d_accumulator.append(d)
            t = takes_many(a, b, c, d)
            sums.append(t)
    return b_accumulator, c_accumulator, d_accumulator, sums


takes_many_node = atomic_model.AtomicNode.model_validate(
    {
        "type": "atomic",
        "fully_qualified_name": f"{takes_many.__module__}.{takes_many.__qualname__}",
        "inputs": ["a", "b", "c", "d"],
        "outputs": ["total"],
    }
)

# In principle, when it's the body is a single node we could remove this layer
# but for now let's keep uniform treatment rather than minimal representation
zbat_for_body = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["a", "b", "c", "d"],
        "outputs": ["t"],
        "nodes": {"takes_many_0": takes_many_node},
        "input_edges": {
            "takes_many_0.a": "a",
            "takes_many_0.b": "b",
            "takes_many_0.c": "c",
            "takes_many_0.d": "d",
        },
        "edges": {},
        "output_edges": {"t": "takes_many_0.total"},
    }
)

zbat_for_node = for_model.ForNode.model_validate(
    {
        "type": "for",
        "inputs": ["a", "bs", "cs", "ds"],
        "outputs": [
            "b_accumulator",
            "c_accumulator",
            "d_accumulator",
            "sums",
        ],
        "body_node": {"label": "body", "node": zbat_for_body},
        "input_edges": {
            "body.a": "a",
            "body.b": "bs",
            "body.c": "cs",
            "body.d": "ds",
        },
        "output_edges": {"sums": "body.t"},
        "nested_ports": ["b"],
        "zipped_ports": ["c", "d"],
        "transfer_edges": {
            "b_accumulator": "bs",
            "c_accumulator": "cs",
            "d_accumulator": "ds",
        },
    }
)

zbat_wf_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["a", "bs", "cs", "ds"],
        "outputs": ["b_accumulator", "c_accumulator", "d_accumulator", "sums"],
        "nodes": {"for_0": zbat_for_node},
        "input_edges": {
            "for_0.a": "a",
            "for_0.bs": "bs",
            "for_0.cs": "cs",
            "for_0.ds": "ds",
        },
        "edges": {},
        "output_edges": {
            "b_accumulator": "for_0.b_accumulator",
            "c_accumulator": "for_0.c_accumulator",
            "d_accumulator": "for_0.d_accumulator",
            "sums": "for_0.sums",
        },
    }
)


def sum_elements(lst: list[int]) -> int:
    total = sum(lst)
    return total


def my_square(n: int) -> int:
    n_sq = n * n
    return n_sq


def nested(ns):
    """
    And here is an example of where we actually use a nested loop.
    Note that each references an accumulator from the same scope and
    appends to it in the body.
    """

    sq_sums = []
    for n in ns:

        rs = my_range(n)
        squares = []
        for r in rs:
            sq = my_square(r)
            squares.append(sq)
        summed = sum_elements(squares)
        sq_sums.append(summed)

    return sq_sums


sum_node = atomic_model.AtomicNode.model_validate(
    {
        "type": "atomic",
        "fully_qualified_name": f"{sum_elements.__module__}.{sum_elements.__qualname__}",
        "inputs": ["lst"],
        "outputs": ["total"],
    }
)

square_node = atomic_model.AtomicNode.model_validate(
    {
        "type": "atomic",
        "fully_qualified_name": f"{my_square.__module__}.{my_square.__qualname__}",
        "inputs": ["n"],
        "outputs": ["n_sq"],
    }
)

# at this point, the parser is fully-featured, so we can use its own output to generate
# the reference and then proof-read it
nested_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["ns"],
        "outputs": ["sq_sums"],
        "nodes": {
            "for_0": {
                "type": "for",
                "inputs": ["ns"],
                "outputs": ["sq_sums"],
                "body_node": {
                    "label": "body",
                    "node": {
                        "type": "workflow",
                        "inputs": ["n"],
                        "outputs": ["summed"],
                        "nodes": {
                            "my_range_0": range_node,
                            "for_0": {
                                "type": "for",
                                "inputs": ["rs"],
                                "outputs": ["squares"],
                                "body_node": {
                                    "label": "body",
                                    "node": {
                                        "type": "workflow",
                                        "inputs": ["r"],
                                        "outputs": ["sq"],
                                        "nodes": {
                                            "my_square_0": square_node,
                                        },
                                        "input_edges": {"my_square_0.n": "r"},
                                        "edges": {},
                                        "output_edges": {"sq": "my_square_0.n_sq"},
                                    },
                                },
                                "input_edges": {"body.r": "rs"},
                                "output_edges": {"squares": "body.sq"},
                                "nested_ports": ["r"],
                                "zipped_ports": [],
                                "transfer_edges": {},
                            },
                            "sum_elements_0": sum_node,
                        },
                        "input_edges": {"my_range_0.n": "n"},
                        "edges": {
                            "for_0.rs": "my_range_0.output_0",
                            "sum_elements_0.lst": "for_0.squares",
                        },
                        "output_edges": {"summed": "sum_elements_0.total"},
                    },
                },
                "input_edges": {"body.n": "ns"},
                "output_edges": {"sq_sums": "body.summed"},
                "nested_ports": ["n"],
                "zipped_ports": [],
                "transfer_edges": {},
            }
        },
        "input_edges": {"for_0.ns": "ns"},
        "edges": {},
        "output_edges": {"sq_sums": "for_0.sq_sums"},
    }
)


def my_offset_range(n: int, offset: int) -> list[int]:
    return list(range(n + offset))


def my_offset_square(n: int, offset: int) -> int:
    n_sq = (n + offset) ** 2
    return n_sq


def nested_with_passed_input(ns, range_offset, square_offset):
    """
    While the accumulators must come from the local scope, note that
    we can safely leverage non-accumulator symbols from arbitrary scope,
    since these can be consistently passed through control flow structures
    """

    sq_sums = []
    for n in ns:

        rs = my_offset_range(n, range_offset)
        squares = []
        for r in rs:
            sq = my_offset_square(r, square_offset)
            # Uses square_offset from the outer context
            squares.append(sq)
        summed = sum_elements(squares)
        sq_sums.append(summed)

    return sq_sums


offset_range_node = atomic_model.AtomicNode.model_validate(
    {
        "type": "atomic",
        "fully_qualified_name": f"{my_offset_range.__module__}.{my_offset_range.__qualname__}",
        "inputs": ["n", "offset"],
        "outputs": ["output_0"],
    }
)

offset_square_node = atomic_model.AtomicNode.model_validate(
    {
        "type": "atomic",
        "fully_qualified_name": f"{my_offset_square.__module__}.{my_offset_square.__qualname__}",
        "inputs": ["n", "offset"],
        "outputs": ["n_sq"],
    }
)

nested_with_passed_input_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["ns", "range_offset", "square_offset"],
        "outputs": ["sq_sums"],
        "nodes": {
            "for_0": {
                "type": "for",
                "inputs": ["range_offset", "square_offset", "ns"],
                "outputs": ["sq_sums"],
                "body_node": {
                    "label": "body",
                    "node": {
                        "type": "workflow",
                        "inputs": ["n", "range_offset", "square_offset"],
                        "outputs": ["summed"],
                        "nodes": {
                            "my_offset_range_0": offset_range_node,
                            "for_0": {
                                "type": "for",
                                "inputs": ["square_offset", "rs"],
                                "outputs": ["squares"],
                                "body_node": {
                                    "label": "body",
                                    "node": {
                                        "type": "workflow",
                                        "inputs": ["r", "square_offset"],
                                        "outputs": ["sq"],
                                        "nodes": {
                                            "my_offset_square_0": offset_square_node,
                                        },
                                        "input_edges": {
                                            "my_offset_square_0.n": "r",
                                            "my_offset_square_0.offset": "square_offset",
                                        },
                                        "edges": {},
                                        "output_edges": {
                                            "sq": "my_offset_square_0.n_sq"
                                        },
                                    },
                                },
                                "input_edges": {
                                    "body.square_offset": "square_offset",
                                    "body.r": "rs",
                                },
                                "output_edges": {"squares": "body.sq"},
                                "nested_ports": ["r"],
                                "zipped_ports": [],
                                "transfer_edges": {},
                            },
                            "sum_elements_0": sum_node,
                        },
                        "input_edges": {
                            "my_offset_range_0.n": "n",
                            "my_offset_range_0.offset": "range_offset",
                            "for_0.square_offset": "square_offset",
                        },
                        "edges": {
                            "for_0.rs": "my_offset_range_0.output_0",
                            "sum_elements_0.lst": "for_0.squares",
                        },
                        "output_edges": {"summed": "sum_elements_0.total"},
                    },
                },
                "input_edges": {
                    "body.range_offset": "range_offset",
                    "body.square_offset": "square_offset",
                    "body.n": "ns",
                },
                "output_edges": {"sq_sums": "body.summed"},
                "nested_ports": ["n"],
                "zipped_ports": [],
                "transfer_edges": {},
            }
        },
        "input_edges": {
            "for_0.range_offset": "range_offset",
            "for_0.square_offset": "square_offset",
            "for_0.ns": "ns",
        },
        "edges": {},
        "output_edges": {"sq_sums": "for_0.sq_sums"},
    }
)


def _field_differences(
    reference: workflow_model.WorkflowNode, actual: workflow_model.WorkflowNode
) -> dict:
    dict1 = reference.model_dump(mode="json")
    dict2 = actual.model_dump(mode="json")
    return {
        k: (dict1.get(k), dict2.get(k))
        for k in dict1.keys() | dict2.keys()
        if dict1.get(k) != dict2.get(k)
    }


class TestParsingForLoops(unittest.TestCase):
    """
    These are pretty brittle to source code changes, as the reference is using the
    fully-stringified nested dictionary representation.
    """

    def test_against_static_recipes(self):
        for function, reference in (
            (single_loop, single_loop_node),
            (zipped_broadcast_and_transferred, zbat_wf_node),
            (nested, nested_node),
            (nested_with_passed_input, nested_with_passed_input_node),
        ):
            with self.subTest(function=function.__name__):
                parsed_node = workflow_parser.parse_workflow(function)
                self.assertEqual(
                    parsed_node,
                    reference,
                    msg=f"Differences: {_field_differences(reference, parsed_node)}",
                )

    def test_custom(self):
        parsed_node = workflow_parser.parse_workflow(nested_with_passed_input)
        print(parsed_node.model_dump(mode="json"))
        self.assertEqual(parsed_node, nested_with_passed_input_node)
