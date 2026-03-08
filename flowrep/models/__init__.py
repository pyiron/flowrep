"""
Use decorators (@atomic and @workflow) to attach a recipe as a `.flowrep_recipe`
attribute onto a function at definition time, and parsers (`parse_atomic`,
`parse_workflow`) to return a recipe from an existing function object.
"""

from flowrep.models.api.parsers import atomic as atomic
from flowrep.models.api.parsers import parse_atomic as parse_atomic
from flowrep.models.api.parsers import parse_workflow as parse_workflow
from flowrep.models.api.parsers import workflow as workflow
