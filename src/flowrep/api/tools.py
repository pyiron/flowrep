"""
Functions for creating and transforming flowrep data.

Intended for users who want to go beyond the simple parsers and decorators, as well as
for downstream library developers.

Use `flowrep2pwd` and `pwd2flowrep` to convert back and forth between flowrep recipes
and the python-workflow-definition format. (When PWD is available in the environment;
Where PWD supports the workflow structure; round-trip from flowrep to PWD and back is
lossy in versioning info.)

Use `recipe2live` to convert recipes into state-ful data-ful instances of recipes,
which directly hold python objects (including, potentially, IO data values), but
which no longer trivially serialize to JSON.

Use decorators (`@atomic` and `@workflow`) to attach a recipe as a `.flowrep_recipe`
attribute onto a function at definition time, and parsers (`parse_atomic`,
`parse_workflow`) to return a recipe from an existing function object.

Use `run_recipe` to convert recipes into live objects with output data.
"""

from flowrep.converters.python_workflow_definition import flowrep2pwd as flowrep2pwd
from flowrep.converters.python_workflow_definition import pwd2flowrep as pwd2flowrep
from flowrep.live import recipe2live as recipe2live
from flowrep.parsers.atomic_parser import atomic as atomic
from flowrep.parsers.atomic_parser import parse_atomic as parse_atomic
from flowrep.parsers.workflow_parser import parse_workflow as parse_workflow
from flowrep.parsers.workflow_parser import workflow as workflow
from flowrep.wfms import run_recipe as run_recipe
