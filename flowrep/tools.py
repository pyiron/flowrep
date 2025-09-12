from collections import defaultdict


def recursive_defaultdict() -> defaultdict:
    """
    Create a recursively nested defaultdict.

    Example:
    >>> d = recursive_defaultdict()
    >>> d['a']['b']['c'] = 1
    >>> print(d)

    Output: 1
    """
    return defaultdict(recursive_defaultdict)


def dict_to_recursive_dd(d: dict | defaultdict) -> defaultdict:
    """Convert a regular dict to a recursively nested defaultdict."""
    if isinstance(d, dict) and not isinstance(d, defaultdict):
        return defaultdict(
            recursive_defaultdict, {k: dict_to_recursive_dd(v) for k, v in d.items()}
        )
    return d


def recursive_dd_to_dict(d: dict | defaultdict) -> dict:
    """Convert a recursively nested defaultdict to a regular dict."""
    if isinstance(d, defaultdict):
        return {k: recursive_dd_to_dict(v) for k, v in d.items()}
    return d
