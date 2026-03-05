"""
Use decorators (@atomic and @workflow) to attach a recipe as a `.flowrep_recipe`
attribute onto a function at definition time, and parsers (`parse_atomic`,
`parse_workflow`) to return a recipe from an existing function object.
"""

from flowrep.models.parsers.atomic_parser import atomic as atomic
from flowrep.models.parsers.atomic_parser import parse_atomic as parse_atomic
from flowrep.models.parsers.workflow_parser import parse_workflow as parse_workflow
from flowrep.models.parsers.workflow_parser import workflow as workflow
