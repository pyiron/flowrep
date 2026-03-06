"""
Bidirectional converter between flowrep and python-workflow-definition formats.

The ``python_workflow_definition`` (pwd) package is an **optional** dependency.
It represents workflows as flat, non-nested DAGs of atomic function calls with
explicit input/output nodes carrying JSON-serializable default values.
"""

from __future__ import annotations

from typing import Any

from pyiron_snippets import import_alarm, versions

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import atomic_model, workflow_model
from flowrep.models.parsers import label_helpers

with import_alarm.ImportAlarm(
    "This converter requires the 'python-workflow-definition' package. "
) as _import_alarm:
    from python_workflow_definition import __version__ as pwd_version
    from python_workflow_definition import models as pwd

# pwd encodes a single unnamed output as ``sourcePort: null``; its validator
# maps that to :data:`pwd.INTERNAL_DEFAULT_HANDLE` (``"__result__"``).
# We reuse the same string as a flowrep output-port name so that the
# representation round-trips without ambiguity.
_DEFAULT_OUTPUT_PORT: str = "__result__"

# Port names in pwd edges may not be valid Python identifiers (e.g. ``"0"``
# used by ``python_workflow_definition.shared.get_list``).  Flowrep requires
# all port names to satisfy :func:`str.isidentifier`, not be keywords, and not
# be in :data:`base_models.RESERVED_NAMES`.  We sanitize on the way in and
# reverse on the way out with a prefix that is itself a valid identifier start.
_PORT_SANITIZE_PREFIX: str = "flowrep_sanitized_"


def _needs_sanitization(port: str) -> bool:
    """Return ``True`` if *port* is not a valid flowrep :class:`Label`."""
    return not base_models.is_valid_label(port)


def _sanitize_port(port: str) -> str:
    """
    Make a pwd port name safe for use as a flowrep :class:`Label`.

    The transformation is reversible via :func:`_desanitize_port` as long as
    no *user-created* port name starts with :data:`_PORT_SANITIZE_PREFIX` and
    has a remainder that would itself need sanitisation.
    """
    if _needs_sanitization(port):
        return _PORT_SANITIZE_PREFIX + port
    return port


def _desanitize_port(port: str) -> str:
    """Reverse :func:`_sanitize_port` when converting back to pwd."""
    if port.startswith(_PORT_SANITIZE_PREFIX):
        candidate = port[len(_PORT_SANITIZE_PREFIX) :]
        if candidate and _needs_sanitization(candidate):
            return candidate
    return port


@_import_alarm
def pwd2flowrep(
    wf: pwd.PythonWorkflowDefinitionWorkflow,
) -> tuple[workflow_model.WorkflowNode, dict[str, pwd.AllowableDefaults]]:
    """
    Convert a *python-workflow-definition* workflow to flowrep.

    Args:
        wf: A validated PWD workflow instance.

    Returns:
        A ``(WorkflowNode, defaults)`` pair where *defaults* maps each
        workflow-input name to the default value carried by the corresponding
        PWD input node.
    """
    input_nodes, output_nodes, function_nodes = _categorize_pwd_nodes(wf.nodes)

    label_map = _build_label_map(function_nodes)
    node_inputs, node_outputs = _collect_function_node_ports(
        wf.edges,
        function_nodes,
    )

    nodes = _build_atomic_nodes(function_nodes, label_map, node_inputs, node_outputs)

    wf_inputs = [input_nodes[nid].name for nid in sorted(input_nodes)]
    wf_outputs = [output_nodes[nid].name for nid in sorted(output_nodes)]
    defaults: dict[str, pwd.AllowableDefaults] = {
        n.name: n.value for n in input_nodes.values()
    }

    fr_input_edges, fr_edges, fr_output_edges = _build_flowrep_edges(
        wf.edges,
        input_nodes,
        output_nodes,
        label_map,
    )

    result = workflow_model.WorkflowNode(
        inputs=wf_inputs,
        outputs=wf_outputs,
        nodes=nodes,
        input_edges=fr_input_edges,
        edges=fr_edges,
        output_edges=fr_output_edges,
    )
    return result, defaults


@_import_alarm
def flowrep2pwd(
    wf: workflow_model.WorkflowNode,
    **terminal_inputs: pwd.AllowableDefaults,
) -> pwd.PythonWorkflowDefinitionWorkflow:
    """
    Convert a flowrep :class:`WorkflowNode` to *python-workflow-definition*.

    Every child of *wf* must be an :class:`AtomicNode` (the pwd format does not
    support nested sub-graphs).  A default value must be supplied for **every**
    workflow input via *terminal_inputs*.

    Args:
        wf: A flat flowrep workflow (atomic children only).
        **terminal_inputs: One keyword argument per workflow input, providing
            the JSON-serialisable default value for the corresponding PWD
            input node.

    Returns:
        A validated :class:`PythonWorkflowDefinitionWorkflow`.

    Raises:
        ValueError: If any child is non-atomic, or if *terminal_inputs* does
            not exactly cover the workflow's inputs.
    """
    _validate_flat_workflow(wf)
    _validate_terminal_inputs(wf, terminal_inputs)

    id_counter = _IdCounter()

    # --- input nodes ---
    input_node_ids: dict[str, int] = {}
    pwd_nodes: list[pwd.PythonWorkflowDefinitionNode] = []
    for port in wf.inputs:
        nid = id_counter.next()
        input_node_ids[port] = nid
        pwd_nodes.append(
            pwd.PythonWorkflowDefinitionInputNode(
                id=nid,
                type="input",
                name=port,
                value=terminal_inputs[port],
            )
        )

    # --- output nodes ---
    output_node_ids: dict[str, int] = {}
    for port in wf.outputs:
        nid = id_counter.next()
        output_node_ids[port] = nid
        pwd_nodes.append(
            pwd.PythonWorkflowDefinitionOutputNode(
                id=nid,
                type="output",
                name=port,
            )
        )

    # --- function nodes ---
    func_node_ids: dict[str, int] = {}
    for label, node in wf.nodes.items():
        nid = id_counter.next()
        func_node_ids[label] = nid
        # Guaranteed by _validate_flat_workflow
        assert isinstance(node, atomic_model.AtomicNode)
        pwd_nodes.append(
            pwd.PythonWorkflowDefinitionFunctionNode(
                id=nid,
                type="function",
                value=node.fully_qualified_name,
            )
        )

    # --- edges ---
    pwd_edges = _build_pwd_edges(wf, input_node_ids, output_node_ids, func_node_ids)

    return pwd.PythonWorkflowDefinitionWorkflow(
        version=pwd_version,
        nodes=pwd_nodes,
        edges=pwd_edges,
    )


def _categorize_pwd_nodes(
    nodes: list[pwd.PythonWorkflowDefinitionNode],
) -> tuple[
    dict[int, pwd.PythonWorkflowDefinitionInputNode],
    dict[int, pwd.PythonWorkflowDefinitionOutputNode],
    dict[int, pwd.PythonWorkflowDefinitionFunctionNode],
]:
    """Partition PWD nodes into input / output / function dicts keyed by id."""
    input_nodes: dict[int, pwd.PythonWorkflowDefinitionInputNode] = {}
    output_nodes: dict[int, pwd.PythonWorkflowDefinitionOutputNode] = {}
    function_nodes: dict[int, pwd.PythonWorkflowDefinitionFunctionNode] = {}
    for node in nodes:
        if isinstance(node, pwd.PythonWorkflowDefinitionInputNode):
            input_nodes[node.id] = node
        elif isinstance(node, pwd.PythonWorkflowDefinitionOutputNode):
            output_nodes[node.id] = node
        elif isinstance(node, pwd.PythonWorkflowDefinitionFunctionNode):
            function_nodes[node.id] = node
    return input_nodes, output_nodes, function_nodes


def _build_label_map(
    function_nodes: dict[int, pwd.PythonWorkflowDefinitionFunctionNode],
) -> dict[int, str]:
    """
    Assign unique string labels to function nodes, deterministic by ID order.

    Uses the last dot-segment of the node's ``value`` (the function name) as
    the base, appended with an incrementing suffix — matching the convention
    used by :func:`flowrep.models.parsers.label_helpers.unique_suffix`.
    """
    counts: dict[str, int] = {}
    label_map: dict[int, str] = {}
    for node_id in sorted(function_nodes):
        fn_node = function_nodes[node_id]
        base = fn_node.value.rsplit(".", 1)[-1]
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        label_map[node_id] = label_helpers.index_label(base, idx)
    return label_map


def _resolve_source_port(port: str | None) -> str:
    """Map pwd's internal sourcePort to a flowrep output-port name."""
    if port is None or port == pwd.INTERNAL_DEFAULT_HANDLE:
        return _DEFAULT_OUTPUT_PORT
    return _sanitize_port(port)


def _collect_function_node_ports(
    edges: list[pwd.PythonWorkflowDefinitionEdge],
    function_nodes: dict[int, pwd.PythonWorkflowDefinitionFunctionNode],
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """
    Collect ordered, unique input/output port names for each function node.

    Port names that are not valid flowrep Labels are sanitized via
    :func:`_sanitize_port`.
    """
    node_inputs: dict[int, list[str]] = {nid: [] for nid in function_nodes}
    node_outputs: dict[int, list[str]] = {nid: [] for nid in function_nodes}

    for edge in edges:
        # Target ports (inputs of the function node)
        if edge.target in function_nodes and edge.targetPort is not None:
            port = _sanitize_port(edge.targetPort)
            ports = node_inputs[edge.target]
            if port not in ports:
                ports.append(port)

        # Source ports (outputs of the function node)
        if edge.source in function_nodes:
            port = _resolve_source_port(edge.sourcePort)
            ports = node_outputs[edge.source]
            if port not in ports:
                ports.append(port)

    return node_inputs, node_outputs


def _build_atomic_nodes(
    function_nodes: dict[int, pwd.PythonWorkflowDefinitionFunctionNode],
    label_map: dict[int, str],
    node_inputs: dict[int, list[str]],
    node_outputs: dict[int, list[str]],
) -> dict[str, atomic_model.AtomicNode]:
    """Build flowrep :class:`AtomicNode` instances from PWD function nodes."""
    nodes: dict[str, atomic_model.AtomicNode] = {}
    for node_id in sorted(function_nodes):
        fn_node = function_nodes[node_id]
        label = label_map[node_id]
        module, qualname = fn_node.value.rsplit(".", 1)
        # This could, in principle, give the incorrect division into module and qualname
        # For the purpose of a fully qualified name, it makes no difference
        # But for someone trying to use just one component it might.
        # Ultimately, we don't have much choice shy of trying to import and recover the
        # underlying object, since PWD itself entangles the two sub-fields.
        # This rsplit is at least consistent with the assumptions made in the PWD
        # ecosystem:
        # https://github.com/pythonworkflow/python-workflow-definition/blob/a372769e190176fe49e71740ad899937df0eeb94/src/python_workflow_definition/purepython.py#L80-L82
        nodes[label] = atomic_model.AtomicNode(
            reference=base_models.PythonReference(
                info=versions.VersionInfo(
                    module=module,
                    qualname=qualname,
                    version=None,  # PWD stores no information for this field
                ),
            ),
            inputs=node_inputs[node_id],
            outputs=node_outputs[node_id],
        )
    return nodes


def _build_flowrep_edges(
    edges: list[pwd.PythonWorkflowDefinitionEdge],
    input_nodes: dict[int, pwd.PythonWorkflowDefinitionInputNode],
    output_nodes: dict[int, pwd.PythonWorkflowDefinitionOutputNode],
    label_map: dict[int, str],
) -> tuple[edge_models.InputEdges, edge_models.Edges, edge_models.OutputEdges]:
    """Convert PWD edges into the three flowrep edge dictionaries."""
    fr_input_edges: edge_models.InputEdges = {}
    fr_edges: edge_models.Edges = {}
    fr_output_edges: edge_models.OutputEdges = {}

    for edge in edges:
        source_is_input = edge.source in input_nodes
        target_is_output = edge.target in output_nodes

        if source_is_input and target_is_output:
            # Pass-through: workflow input → workflow output
            in_name = input_nodes[edge.source].name
            out_name = output_nodes[edge.target].name
            fr_output_edges[edge_models.OutputTarget(port=out_name)] = (
                edge_models.InputSource(port=in_name)
            )

        elif source_is_input:
            # Workflow input → function node
            in_name = input_nodes[edge.source].name
            target_label = label_map[edge.target]
            target_port = _sanitize_port(edge.targetPort)
            fr_input_edges[
                edge_models.TargetHandle(node=target_label, port=target_port)
            ] = edge_models.InputSource(port=in_name)

        elif target_is_output:
            # Function node → workflow output
            out_name = output_nodes[edge.target].name
            source_label = label_map[edge.source]
            source_port = _resolve_source_port(edge.sourcePort)
            fr_output_edges[edge_models.OutputTarget(port=out_name)] = (
                edge_models.SourceHandle(node=source_label, port=source_port)
            )

        else:
            # Function node → function node (sibling edge)
            source_label = label_map[edge.source]
            target_label = label_map[edge.target]
            source_port = _resolve_source_port(edge.sourcePort)
            target_port = _sanitize_port(edge.targetPort)
            fr_edges[edge_models.TargetHandle(node=target_label, port=target_port)] = (
                edge_models.SourceHandle(node=source_label, port=source_port)
            )

    return fr_input_edges, fr_edges, fr_output_edges


class _IdCounter:
    """Simple incrementing integer-ID generator."""

    def __init__(self, start: int = 0) -> None:
        self._next = start

    def next(self) -> int:
        nid = self._next
        self._next += 1
        return nid


def _validate_flat_workflow(wf: workflow_model.WorkflowNode) -> None:
    """Raise :class:`ValueError` if any child is not an :class:`AtomicNode`."""
    for label, node in wf.nodes.items():
        if not isinstance(node, atomic_model.AtomicNode):
            raise ValueError(
                f"flowrep2pwd requires all children to be AtomicNode, but "
                f"'{label}' is {type(node).__name__}."
            )


def _validate_terminal_inputs(
    wf: workflow_model.WorkflowNode,
    terminal_inputs: dict[str, Any],
) -> None:
    """Raise :class:`ValueError` if *terminal_inputs* != workflow inputs."""
    expected = set(wf.inputs)
    provided = set(terminal_inputs)
    missing = expected - provided
    extra = provided - expected
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"extra: {sorted(extra)}")
        raise ValueError(
            f"terminal_inputs must exactly match workflow inputs. {'; '.join(parts)}"
        )


def _flowrep_port_to_pwd_source_port(port: str) -> str | None:
    """
    Map a flowrep output-port name back to a pwd ``sourcePort`` value.

    Returns ``None`` for the default-output sentinel so that the PWD edge
    validator stores it as :data:`pwd.INTERNAL_DEFAULT_HANDLE`.  Otherwise
    reverses any sanitisation applied by :func:`_sanitize_port`.
    """
    if port == _DEFAULT_OUTPUT_PORT:
        return None
    return _desanitize_port(port)


def _build_pwd_edges(
    wf: workflow_model.WorkflowNode,
    input_node_ids: dict[str, int],
    output_node_ids: dict[str, int],
    func_node_ids: dict[str, int],
) -> list[pwd.PythonWorkflowDefinitionEdge]:
    """
    Convert the three flowrep edge dicts into a flat PWD edge list.

    Edges are emitted in a deterministic order that preserves the input-port
    ordering of each child node — this is important for consumers that rely on
    edge-list order (e.g. ``get_list``).
    """
    pwd_edges: list[pwd.PythonWorkflowDefinitionEdge] = []

    # Emit edges targeting each child, one per input port in order.
    # A given TargetHandle appears in exactly one of input_edges or edges.
    source: edge_models.SourceHandle | edge_models.InputSource
    for label, node in wf.nodes.items():
        for port in node.inputs:
            th = edge_models.TargetHandle(node=label, port=port)
            target_port_pwd = _desanitize_port(port)

            if th in wf.input_edges:
                source = wf.input_edges[th]
                pwd_edges.append(
                    pwd.PythonWorkflowDefinitionEdge(
                        source=input_node_ids[source.port],
                        sourcePort=None,
                        target=func_node_ids[label],
                        targetPort=target_port_pwd,
                    )
                )
            elif th in wf.edges:
                source = wf.edges[th]
                pwd_edges.append(
                    pwd.PythonWorkflowDefinitionEdge(
                        source=func_node_ids[source.node],
                        sourcePort=_flowrep_port_to_pwd_source_port(source.port),
                        target=func_node_ids[label],
                        targetPort=target_port_pwd,
                    )
                )

    # Emit output edges in workflow output order.
    for port in wf.outputs:
        ot = edge_models.OutputTarget(port=port)
        if ot in wf.output_edges:
            source = wf.output_edges[ot]
            if isinstance(source, edge_models.InputSource):
                pwd_edges.append(
                    pwd.PythonWorkflowDefinitionEdge(
                        source=input_node_ids[source.port],
                        sourcePort=None,
                        target=output_node_ids[port],
                        targetPort=None,
                    )
                )
            else:
                pwd_edges.append(
                    pwd.PythonWorkflowDefinitionEdge(
                        source=func_node_ids[source.node],
                        sourcePort=_flowrep_port_to_pwd_source_port(source.port),
                        target=output_node_ids[port],
                        targetPort=None,
                    )
                )

    return pwd_edges
