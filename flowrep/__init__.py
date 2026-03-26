"""
Use decorators (@atomic and @workflow) to attach a recipe as a `.flowrep_recipe`
attribute onto a function at definition time, and parsers (`parse_atomic`,
`parse_workflow`) to return a recipe from an existing function object.
"""

import importlib.metadata

from flowrep.api.parsers import atomic as atomic
from flowrep.api.parsers import parse_atomic as parse_atomic
from flowrep.api.parsers import parse_workflow as parse_workflow
from flowrep.api.parsers import workflow as workflow

try:
    # Installed package will find its version
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # Repository clones will register an unknown version
    __version__ = "0.0.0+unknown"
