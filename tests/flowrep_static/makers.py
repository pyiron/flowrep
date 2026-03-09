"""Helper functions for common maker patterns in the tests."""

from pyiron_snippets import versions

from flowrep.models import base_models
from flowrep.models.nodes import atomic_model, helper_models


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
