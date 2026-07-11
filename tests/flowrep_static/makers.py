"""Helper functions for common maker patterns in the tests."""

from pyiron_snippets import versions

from flowrep import base_models, edge_models
from flowrep.parsers import workflow_parser
from flowrep.prospective import atomic_recipe, helper_models, workflow_recipe

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
) -> atomic_recipe.AtomicRecipe:
    return atomic_recipe.AtomicRecipe(
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
) -> helper_models.LabeledRecipe:
    return helper_models.LabeledRecipe(
        label=label,
        recipe=make_atomic(
            inputs=inputs or [],
            outputs=outputs or [],
            module=module,
            qualname=qualname,
            version=version,
            inputs_with_defaults=inputs_with_defaults,
        ),
    )


def make_labeled_with_defaults(label: str) -> helper_models.LabeledRecipe:
    """Just add a label and go! IO exists and input has a default."""
    return make_labeled_atomic(
        label,
        inputs=["x"],
        outputs=["y"],
        qualname=label,
        inputs_with_defaults=["x"],
    )


def make_simple_workflow_recipe() -> workflow_recipe.WorkflowRecipe:
    """One-child workflow: ``add(a, b) -> result``."""
    return workflow_recipe.WorkflowRecipe(
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


########################################
# For casting workflows back to python #
########################################


def dump_no_refs(recipe) -> dict:
    """Model dump with every (possibly nested) 'reference' and 'description' removed.

    'reference' is dropped because a reference-free workflow used as a peer node
    necessarily re-parses as a referenced node. 'description' is dropped because a
    docstring round-trips through inspect.getdoc/cleandoc, which is not idempotent
    for bodies indented relative to their first line, so the stored description
    cannot always be reproduced exactly.
    """

    def strip(obj):
        if isinstance(obj, dict):
            return {
                k: strip(v)
                for k, v in obj.items()
                if k not in ("reference", "description")
            }
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    return strip(recipe.model_dump(mode="json"))


def reference_free(func) -> "object":
    """Parse a decorated/plain function and drop the top-level reference."""
    if isinstance(func, workflow_recipe.WorkflowRecipe):
        recipe = func
    else:
        recipe = workflow_parser.parse_workflow(func)
    return recipe.model_copy(update={"reference": None})
