"""
Use decorators (@atomic and @workflow) to attach a recipe as a `.flowrep_recipe`
attribute onto a function at definition time, and parsers (`parse_atomic`,
`parse_workflow`) to return a recipe from an existing function object.
"""

import importlib.metadata

from flowrep.api import accumulator as accumulator
from flowrep.api import atomic as atomic
from flowrep.api import dataclass as dataclass
from flowrep.api import parse_atomic as parse_atomic
from flowrep.api import parse_workflow as parse_workflow
from flowrep.api import schemas as schemas
from flowrep.api import std as std
from flowrep.api import tools as tools
from flowrep.api import workflow as workflow

try:
    # Installed package will find its version
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    # Repository clones will register an unknown version
    __version__ = "0.0.0+unknown"
