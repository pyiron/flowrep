from collections import defaultdict


# Indefinite nesting
def recursive_defaultdict() -> defaultdict:
    return defaultdict(recursive_defaultdict)


# Convert a regular dict -> recursive_defaultdict
def dict_to_recursive_dd(d: dict | defaultdict) -> defaultdict:
    if isinstance(d, dict) and not isinstance(d, defaultdict):
        return defaultdict(
            recursive_defaultdict, {k: dict_to_recursive_dd(v) for k, v in d.items()}
        )
    return d


# Convert recursive_defaultdict -> plain dict
def recursive_dd_to_dict(d: dict | defaultdict) -> dict:
    if isinstance(d, defaultdict):
        return {k: recursive_dd_to_dict(v) for k, v in d.items()}
    return d
