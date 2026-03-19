"""
Data classes for "live" stateful instance-views of the recipes.

Intended to be a common export- and communication-format for WfMS of instance-views.
Since they hold live, arbitrary python objects, they don't serialize trivially and are
not (by themselves) a valid storage format.
"""

from flowrep.models.live import recipe2live as recipe2live
