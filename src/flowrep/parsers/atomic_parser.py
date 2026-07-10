import ast
import dataclasses
import inspect
import types
from collections.abc import Callable, Iterable
from typing import (
    Annotated,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from pyiron_snippets import versions

from flowrep import base_models
from flowrep.parsers import label_helpers, object_scope, parser_helpers
from flowrep.parsers.label_helpers import default_output_label
from flowrep.prospective import atomic_recipe, helper_models

_AtomicTarget = TypeVar("_AtomicTarget", bound=types.FunctionType | type)


@overload
def atomic(func: _AtomicTarget, /) -> _AtomicTarget: ...
@overload
def atomic(
    func: str | None = None,
    /,
    *output_labels: str,
    unpack_mode: atomic_recipe.UnpackMode = atomic_recipe.UnpackMode.TUPLE,
    version_scraping: versions.VersionScrapingMap | None = None,
    forbid_main: bool = False,
    forbid_locals: bool = False,
    require_version: bool = False,
) -> Callable[[_AtomicTarget], _AtomicTarget]: ...
def atomic(func=None, /, *output_labels, **kwargs):
    """
    Decorator that attaches a :class:`~flowrep.models.nodes.atomic_recipe.AtomicRecipe`
    to the ``flowrep_recipe`` attribute of a function.

    The decorated function's module, qualname, and (optionally) package version are
    captured as provenance metadata via
    :meth:`~pyiron_snippets.versions.VersionInfo.of`.

    Can be used with or without arguments.

    Args:
        func: The function to decorate. Passed positionally by Python when the
            decorator is used without parentheses.
        *output_labels: Explicit names for the node's output ports. When provided,
            their count must match the number of outputs inferred from the function
            and the chosen ``unpack_mode``.
        unpack_mode: How to convert the function's return value into output ports.
            See :class:`~flowrep.models.nodes.atomic_recipe.UnpackMode`.
        version_scraping: Optional mapping from top-level package names to callables
            that return a version string, for packages that don't expose
            ``__version__``. Forwarded to
            :meth:`~pyiron_snippets.versions.VersionInfo.of`.
        forbid_main: If ``True``, raise if the function's module is ``__main__``.
        forbid_locals: If ``True``, raise if the function's qualname contains
            ``<locals>`` (i.e. it was defined inside another function).
        require_version: If ``True``, raise if no version can be determined for
            the function's package.

    Returns:
        The original function with a ``flowrep_recipe`` attribute holding an
        :class:`~flowrep.models.nodes.atomic_recipe.AtomicRecipe`.
    """

    def wrap(f, labels):
        f.flowrep_recipe = parse_atomic(f, *labels, **kwargs)
        return f

    return parser_helpers.apply_label_decorator(
        func,
        output_labels,
        wrap=wrap,
        decorator_name="@atomic",
        allowed_types=(types.FunctionType, type),
    )


def parse_atomic(
    func: types.FunctionType | type,
    *output_labels: str,
    unpack_mode: atomic_recipe.UnpackMode = atomic_recipe.UnpackMode.TUPLE,
    version_scraping: versions.VersionScrapingMap | None = None,
    forbid_main: bool = False,
    forbid_locals: bool = False,
    require_version: bool = False,
) -> atomic_recipe.AtomicRecipe:
    """
    Build an :class:`~flowrep.models.nodes.atomic_recipe.AtomicRecipe` from a plain
    Python function.

    Introspects the function to determine its fully qualified name, package version,
    input parameter names, and output port names (via AST return-value analysis and/or
    type annotations).

    Args:
        func: The function to represent as an atomic node.
        *output_labels: Explicit output port names. When provided, their count must
            match the number of outputs inferred from the function and the chosen
            ``unpack_mode``.
        unpack_mode: How to convert the function's return value into output ports.
        version_scraping: Optional version-scraping overrides, forwarded to
            :meth:`~pyiron_snippets.versions.VersionInfo.of`.
        forbid_main: If ``True``, raise if the function's module is ``__main__``.
        forbid_locals: If ``True``, raise if the function's qualname contains
            ``<locals>``.
        require_version: If ``True``, raise if no version can be determined.

    Returns:
        A fully constructed :class:`AtomicRecipe`.

    Raises:
        ValueError: If ``output_labels`` length mismatches the inferred output count,
            or if any ``forbid_*`` / ``require_*`` constraint is violated.
    """
    if isinstance(func, type):
        base_models.ensure_class_signature_from_init(func, "@atomic")

    function_info = versions.VersionInfo.of(
        func,
        version_scraping=version_scraping,
        forbid_main=forbid_main,
        forbid_locals=forbid_locals,
        require_version=require_version,
    )
    sig_info = parser_helpers.SignatureInfo.of(func)
    docstring = inspect.getdoc(func)

    scraped_output_labels = (
        ["instance"]
        if isinstance(func, type)
        else _get_output_labels(func, unpack_mode)
    )
    if len(output_labels) > 0 and len(output_labels) != len(scraped_output_labels):
        raise ValueError(
            "Explicitly provided output labels must match the function analysis and "
            f"unpack_mode: expected {len(scraped_output_labels)} labels for "
            f"unpack_mode='{unpack_mode}', got {len(output_labels)} labels "
            f"{output_labels}; inferred labels were {scraped_output_labels}."
        )

    return atomic_recipe.AtomicRecipe(
        reference=base_models.PythonReference(
            info=function_info,
            inputs_with_defaults=sig_info.have_defaults,
            restricted_input_kinds=sig_info.have_restricted_kinds,
        ),
        inputs=sig_info.names,
        outputs=(
            list(output_labels) if len(output_labels) > 0 else scraped_output_labels
        ),
        description=docstring,
        unpack_mode=unpack_mode,
    )


def _get_output_labels(
    func: types.FunctionType, unpack_mode: atomic_recipe.UnpackMode
) -> list[str]:
    if unpack_mode == atomic_recipe.UnpackMode.NONE:
        return _parse_return_label_without_unpacking(func)
    elif unpack_mode == atomic_recipe.UnpackMode.TUPLE:
        return _parse_tuple_return_labels(func)
    elif unpack_mode == atomic_recipe.UnpackMode.DATACLASS:
        return _parse_dataclass_return_labels(func)
    raise TypeError(
        f"Invalid unpack mode: {unpack_mode}. Possible values are "
        f"{', '.join(atomic_recipe.UnpackMode.__members__.values())}"
    )


def _parse_return_label_without_unpacking(func: types.FunctionType) -> list[str]:
    """
    Get output label for UnpackMode.NONE.

    Looks for annotation on the return type itself (not tuple elements).
    For `-> Annotated[T, {"label": "x"}]` or `-> Annotated[tuple[...], {"label": "x"}]`
    """
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        return [label_helpers.default_output_label(0)]

    return_hint = hints.get("return")
    if return_hint is None:
        return [label_helpers.default_output_label(0)]

    # Extract label from the outermost Annotated wrapper
    label = label_helpers.extract_label_from_annotated(return_hint)
    return [label] if label is not None else [label_helpers.default_output_label(0)]


def _parse_tuple_return_labels(func: types.FunctionType) -> list[str]:
    func_node = parser_helpers.get_ast_function_node(func)
    return_labels = _extract_combined_return_labels(func_node)
    if not all(len(ret) == len(return_labels[0]) for ret in return_labels):
        raise ValueError(
            f"All return statements must have the same number of elements, got "
            f"{return_labels}"
        )

    # Get AST-scraped labels
    scraped = list(
        (
            label
            if all(other_branch[i] == label for other_branch in return_labels)
            else label_helpers.default_output_label(i)
        )
        for i, label in enumerate(return_labels[0])
    )

    # Override with annotation-based labels where available
    annotated = label_helpers.get_annotated_output_labels(func)
    return label_helpers.merge_labels(
        first_choice=annotated,
        fallback=scraped,
        message_prefix="Annotations and scraped return labels mis-match. ",
    )


def _extract_combined_return_labels(
    func_node: ast.FunctionDef,
) -> list[tuple[str, ...]]:
    return_stmts = [n for n in ast.walk(func_node) if isinstance(n, ast.Return)]
    return_labels: list[tuple[str, ...]] = [()] if len(return_stmts) == 0 else []
    for ret in return_stmts:
        return_labels.append(_extract_return_labels(ret))
    return return_labels


def _extract_return_labels(ret: ast.Return) -> tuple[str, ...]:
    if ret.value is None:
        return_labels: tuple[str, ...] = ()
        return return_labels
    elif isinstance(ret.value, ast.Tuple):
        return tuple(
            elt.id if isinstance(elt, ast.Name) else default_output_label(i)
            for i, elt in enumerate(ret.value.elts)
        )
    else:
        return (
            (ret.value.id,)
            if isinstance(ret.value, ast.Name)
            else (default_output_label(0),)
        )


def _parse_dataclass_return_labels(func: types.FunctionType) -> list[str]:
    source_code_return = _parse_tuple_return_labels(func)
    if len(source_code_return) != 1:
        raise ValueError(
            f"Dataclass unpack mode requires function code to returns to consist of "
            f"exactly one value, i.e. the dataclass instance, but got "
            f"{source_code_return}"
        )

    try:
        hints = get_type_hints(func, include_extras=True)
    except NameError as e:
        raise NameError(
            "Dataclass unpack mode requires the return annotation to be importable at "
            "runtime. Evaluating the return annotation for "
            f"{func.__module__}.{func.__qualname__} failed with: {e}"
        ) from e
    ann = hints.get("return")

    origin = get_origin(ann)
    return_hint = get_args(ann)[0] if origin is Annotated else ann

    if dataclasses.is_dataclass(return_hint):
        return [f.name for f in dataclasses.fields(return_hint)]

    raise ValueError(
        f"Dataclass unpack mode requires a return type annotation that is a "
        f"(perhaps Annotated) dataclass, but got {return_hint}"
    )


def get_labeled_recipe(
    ast_call: ast.Call,
    existing_names: Iterable[str],
    scope: object_scope.ScopeProxy,
    info_factory: versions.VersionInfoFactory,
) -> helper_models.LabeledRecipe:
    child_call = object_scope.resolve_symbol_to_object(ast_call.func, scope)
    if isinstance(child_call, base_models.NodeRecipe):
        child_recipe = child_call
        label_prefix = _infer_node_name(child_recipe, ast_call.func)
    else:
        # Otherwise we're going to find it has already been parsed as a recipe,
        # or we're going to parse it as a recipe -- either way, it had better be a
        # types.FunctionType!
        function_call = cast(types.FunctionType, child_call)
        label_prefix = function_call.__name__
        if hasattr(function_call, "flowrep_recipe"):
            child_recipe = function_call.flowrep_recipe
            if hasattr(child_recipe, "reference") and isinstance(
                child_recipe.reference.info, versions.VersionInfo
            ):
                child_recipe.reference.info.validate_constraints(
                    forbid_main=info_factory.forbid_main,
                    forbid_locals=info_factory.forbid_locals,
                    require_version=info_factory.require_version,
                )
        else:
            child_recipe = parse_atomic(
                function_call,
                version_scraping=info_factory.version_scraping,
                forbid_main=info_factory.forbid_main,
                forbid_locals=info_factory.forbid_locals,
                require_version=info_factory.require_version,
            )
    label = label_helpers.unique_suffix(label_prefix, existing_names)
    return helper_models.LabeledRecipe(label=label, node=child_recipe)


def _infer_node_name(node: base_models.NodeRecipe, ast_call: ast.expr) -> str:
    reference = getattr(node, "reference", None)
    if reference is not None:
        underlying_function_name = reference.info.qualname.rsplit(".", 1)[-1]
        return underlying_function_name
    elif isinstance(ast_call, ast.Name):
        variable_name = ast_call.id
        return variable_name
    elif isinstance(ast_call, ast.Attribute):
        proximate_attribute_name = ast_call.attr
        return proximate_attribute_name
    else:
        raise ValueError(f"Unexpected node type: {type(ast_call)}")
