from pyiron_snippets import versions

from flowrep.models import base_models


def make_reference(
    module: str = "mod",
    qualname: str = "func",
    version: str | None = None,
    inputs_with_defaults: list[str] | None = None,
) -> base_models.PythonReference:
    return base_models.PythonReference(
        info=versions.VersionInfo(module=module, qualname=qualname, version=version),
        inputs_with_defaults=inputs_with_defaults or [],
    )
