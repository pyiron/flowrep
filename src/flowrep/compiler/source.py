from __future__ import annotations

import dataclasses
import inspect
import linecache
import pathlib
import sys
import types
import typing
import weakref
from typing import Any

from flowrep import base_models, datastructures
from flowrep.compiler import function
from flowrep.prospective import workflow_recipe

# Counter used to make each generated source's linecache key unique.
_GENERATED_COUNTER = 0


def _next_generated_filename(name: str) -> str:
    global _GENERATED_COUNTER
    _GENERATED_COUNTER += 1
    return f"<flowrep_generated_{name}_{_GENERATED_COUNTER}>"


def _purge_generated(mod_name: str, fname: str) -> None:
    """Drop a built function's synthetic module + linecache entries (idempotent)."""
    sys.modules.pop(mod_name, None)
    linecache.cache.pop(fname, None)


class _WeakFnModule(types.ModuleType):
    """A synthetic module that holds its generated function via a weakref.

    ``module.__dict__`` does not hold a strong reference to the built function,
    so the only strong reference is the caller's. When the function is
    garbage-collected the weakref is dead and the finalizer can purge the
    ``sys.modules`` and ``linecache`` entries.

    ``__getattr__`` re-exposes the function (while alive) so that
    ``retrieve.import_from_string`` can resolve ``mod_name.fn_name`` normally.
    """

    def __init__(
        self, name: str, fn_name: str, fn_ref: weakref.ref[types.FunctionType]
    ) -> None:
        super().__init__(name)
        # Store the weakref under a private key so normal attribute lookup
        # still falls through to __getattr__ for the public function name.
        self.__dict__["_fn_name"] = fn_name
        self.__dict__["_fn_ref"] = fn_ref

    def __getattr__(self, name: str) -> object:
        if name == self.__dict__.get("_fn_name"):
            fn = self.__dict__["_fn_ref"]()
            if fn is not None:
                return fn
        raise AttributeError(name)


@dataclasses.dataclass
class RenderedSource:
    """Executable Python source plus the live namespace it must be exec'd against."""

    source: str
    namespace: dict[str, Any]
    function_name: str

    def build(self) -> types.FunctionType:
        """exec the source against a copy of the namespace and return the function."""
        fname = _next_generated_filename(self.function_name)
        # Register source with linecache so inspect.getsource works on the result.
        lines = [line + "\n" for line in self.source.splitlines()]
        linecache.cache[fname] = (len(self.source), None, lines, fname)
        # Create a synthetic module and register it in sys.modules BEFORE exec.
        # This ensures inspect.getmodule() resolves to our module for both the
        # nested-def decoration at exec time and the target re-parse by parse_workflow.
        # The module name is derived from the filename (strip angle brackets).
        mod_name = fname.strip("<>")
        # Use a plain ModuleType for exec so the function lands in the dict normally.
        # After exec we switch sys.modules to a _WeakFnModule that exposes the
        # function via a weakref (so the module dict does not pin it).
        exec_module = types.ModuleType(mod_name)
        exec_module.__file__ = fname
        exec_module.__dict__.update({"typing": typing, **self.namespace})
        sys.modules[mod_name] = exec_module
        code = compile(self.source, fname, "exec")
        exec(
            code, exec_module.__dict__
        )  # noqa: S102 - controlled codegen, not user input
        fn = exec_module.__dict__[self.function_name]
        # The source uses `from __future__ import annotations`, so __annotations__
        # holds PEP-563 strings (e.g. '_ann_x'). Resolve them against the module
        # globals so inspect.signature reports the real type objects. The decorator
        # has already run and re-parsing reads source text, so this is safe.
        fn.__annotations__ = typing.get_type_hints(fn, include_extras=True)
        # Bound the synthetic sys.modules / linecache entries to the built function's
        # lifetime. Replace the exec module in sys.modules with a _WeakFnModule that
        # holds only a weakref to fn, so sys.modules no longer transitively pins it.
        # fn keeps its own __globals__ (exec_module.__dict__) alive via fn.__globals__.
        # The _WeakFnModule re-exposes fn through __getattr__ while it is alive, so
        # retrieve.import_from_string("mod.fn_name") continues to work. When fn is
        # garbage-collected the finalizer purges both sys.modules and linecache entries.
        fn_ref: weakref.ref[types.FunctionType] = weakref.ref(fn)
        weak_module = _WeakFnModule(mod_name, self.function_name, fn_ref)
        weak_module.__file__ = fname
        weak_module.__dict__.update(exec_module.__dict__)
        del weak_module.__dict__[self.function_name]
        sys.modules[mod_name] = weak_module
        # Drop fn from exec_module.__dict__ (which fn.__globals__ aliases) to avoid a
        # reference cycle (fn -> __globals__ -> fn) that would require the cyclic GC to
        # collect it. After this, the only strong reference to fn is the caller's.
        del exec_module.__dict__[self.function_name]
        weakref.finalize(fn, _purge_generated, mod_name, fname)
        return fn

    def dump(
        self,
        path: pathlib.Path | str,
        *,
        parents: bool = True,
        exists_ok: bool = True,
        allow_namespace_symbols: bool = False,
    ) -> str:
        """Write ``self.source`` to ``path`` and return a status message.

        A missing extension is filled in as ``.py``; any other extension is
        rejected. Namespace symbols appear as unbound references in the dumped
        file, so dumping a non-empty namespace requires opting in.

        Raises:
            ValueError: Non-``.py`` extension, or namespace symbols present
                without ``allow_namespace_symbols``.
            FileExistsError: ``path`` exists and ``exists_ok`` is False.
        """
        path = pathlib.Path(path).resolve()

        if not path.suffix:
            path = path.with_suffix(".py")
        elif path.suffix != ".py":
            raise ValueError(f"Expected a .py path, got {path.suffix!r}: {path}")

        if path.exists() and not exists_ok:
            raise FileExistsError(
                f"{path} already exists; pass exists_ok=True to overwrite it."
            )

        if self.namespace and not allow_namespace_symbols:
            raise ValueError(
                f"Source references {len(self.namespace)} namespace symbol(s) that "
                "cannot be resolved from the file alone; pass "
                "allow_namespace_symbols=True to dump anyway."
            )

        if parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source, encoding="utf-8")

        message = f"Dumped source to {path}"
        if allow_namespace_symbols and self.namespace:
            symbols = ", ".join(sorted(self.namespace))
            message += (
                "\nReminder: replace the dumped namespace symbol(s) with real "
                f"values before use: {symbols}"
            )
        return message


def flowrep2python(
    workflow: workflow_recipe.WorkflowRecipe | datastructures.DagData,
    function_name: base_models.Label | None = None,
    signature: inspect.Signature | None = None,
    _workflow_decorator: tuple[str, str] = ("flowrep", "workflow"),
) -> RenderedSource:
    """
    Compile a workflow recipe into a rendered object that can be built into a function
    object or dumped to a .py file.
    """
    if isinstance(workflow, datastructures.DagData):
        if signature is not None:
            raise ValueError(
                "Cannot pass signature when compiling a "
                f"{datastructures.DagData.__name__}"
            )
        return _dagdata2python(
            workflow,
            function_name=function_name,
            _workflow_decorator=_workflow_decorator,
        )
    elif isinstance(workflow, workflow_recipe.WorkflowRecipe):
        return _workflow2python(
            workflow,
            function_name=function_name,
            signature=signature,
            _workflow_decorator=_workflow_decorator,
        )
    else:
        raise TypeError(
            f"Expected a {workflow_recipe.WorkflowRecipe.__name__} or "
            f"{datastructures.DagData.__name__} or DagData, got {type(workflow)}"
        )


def _workflow2python(
    recipe: workflow_recipe.WorkflowRecipe,
    *,
    function_name: base_models.Label | None = None,
    signature: inspect.Signature | None = None,
    _workflow_decorator: tuple[str, str] = ("flowrep", "workflow"),
) -> RenderedSource:
    function_name = (
        "compiled_from_workflow_recipe" if function_name is None else function_name
    )
    if recipe.reference is not None:
        raise ValueError(
            f"This recipe already has an underlying Python reference: "
            f"{recipe.reference}"
        )
    emitter = function.Emitter(workflow_decorator=_workflow_decorator)
    decorator_module, _ = _workflow_decorator
    # Reserve every bare module-level name a nested def must not shadow: the top
    # function name, the always-present decorator/typing imports, and the
    # top-level binding of every import this recipe will emit.
    emitter.module_names.reserve("__future__")
    emitter.module_names.reserve("annotations")
    emitter.module_names.reserve("typing")
    emitter.module_names.reserve(decorator_module)
    emitter.module_names.reserve(function_name)
    for binding in function.referenced_top_level_bindings(recipe):
        emitter.module_names.reserve(binding)
    target = function.emit_workflow_function(recipe, function_name, emitter, signature)
    # All generated imports (per-node call, exception, and annotation) are
    # collected in emitter.module_imports and merged here. The preamble also
    # provides the deferred-annotation flag (so return annotations are not
    # evaluated at exec time) plus `typing` (for user annotations) and `flowrep`
    # (for the @workflow decorator emitted on every function). Hoisting to the
    # module level deduplicates imports across the top-level function and its
    # nested defs, and typing.get_type_hints() resolves them from fn.__globals__.
    modules = sorted({"typing", decorator_module} | emitter.module_imports)
    import_block = "\n".join(f"import {m}" for m in modules)
    preamble = "from __future__ import annotations\n\n" + import_block + "\n"
    nested = "\n".join(emitter.nested_defs)
    func_src = target.render()
    parts = [preamble]
    if nested:
        parts.append(nested)
    parts.append(func_src)
    source = "\n".join(parts) + "\n"
    return RenderedSource(
        source=source, namespace=emitter.namespace, function_name=function_name
    )


def _dagdata2python(
    dagdata: datastructures.DagData,
    *,
    function_name: base_models.Label | None = None,
    _workflow_decorator: tuple[str, str] = ("flowrep", "workflow"),
) -> RenderedSource:
    sig = function.build_signature(dagdata.input_ports, dagdata.output_ports)
    # Strip the reference so recipe2python accepts the recipe.
    free_recipe = dagdata.recipe.model_copy(update={"reference": None})
    return _workflow2python(
        free_recipe,
        function_name=function_name,
        signature=sig,
        _workflow_decorator=_workflow_decorator,
    )
