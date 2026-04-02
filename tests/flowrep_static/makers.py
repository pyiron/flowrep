"""Helper functions for common maker patterns in the tests."""

from pyiron_snippets import versions

from flowrep import base_models, edge_models
from flowrep.nodes import atomic_model, helper_models, workflow_model

from flowrep_static import library


def make_reference(
    module: str = "mod",
    qualname: str = "func",
    version: str | None = None,
    inputs_with_defaults: list[str] | None = None,
    restricted_input_kinds: dict[str, base_models.RestrictedParamKind] | None = None,
) -> base_models.PythonReference:
    return base_models.PythonReference(
        info=versions.VersionInfo(module=module, qualname=qualname, version=version),
        inputs_with_defaults=inputs_with_defaults or [],
        restricted_input_kinds=restricted_input_kinds or {},
    )


def make_atomic(
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    module: str = "mod",
    qualname: str = "func",
    version: str | None = None,
    inputs_with_defaults: list[str] | None = None,
) -> atomic_model.AtomicNode:
    return atomic_model.AtomicNode(
        reference=make_reference(
            module=module,
            qualname=qualname,
            version=version,
            inputs_with_defaults=inputs_with_defaults,
        ),
        inputs=inputs or [],
        outputs=outputs or [],
    )


def make_labeled_atomic(
    label: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    module: str = "mod",
    qualname: str = "func",
    version: str | None = None,
    inputs_with_defaults: list[str] | None = None,
) -> helper_models.LabeledNode:
    return helper_models.LabeledNode(
        label=label,
        node=make_atomic(
            inputs=inputs or [],
            outputs=outputs or [],
            module=module,
            qualname=qualname,
            version=version,
            inputs_with_defaults=inputs_with_defaults,
        ),
    )


def make_labeled_with_defaults(label: str) -> helper_models.LabeledNode:
    """Just add a label and go! IO exists and input has a default."""
    return make_labeled_atomic(
        label,
        inputs=["x"],
        outputs=["y"],
        qualname=label,
        inputs_with_defaults=["x"],
    )


def make_simple_workflow_recipe() -> workflow_model.WorkflowNode:
    """One-child workflow: ``add(a, b) -> result``."""
    return workflow_model.WorkflowNode(
        inputs=["a", "b"],
        outputs=["result"],
        nodes={"add_0": library.my_add.flowrep_recipe},
        input_edges={
            edge_models.TargetHandle(node="add_0", port="a"): edge_models.InputSource(
                port="a"
            ),
            edge_models.TargetHandle(node="add_0", port="b"): edge_models.InputSource(
                port="b"
            ),
        },
        edges={},
        output_edges={
            edge_models.OutputTarget(port="result"): edge_models.SourceHandle(
                node="add_0", port="output_0"
            ),
        },
    )
