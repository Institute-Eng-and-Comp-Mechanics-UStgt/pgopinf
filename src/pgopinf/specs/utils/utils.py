def _as_tuple(x):
    if x is None:
        return tuple()
    if isinstance(x, str):
        return (x,)
    return tuple(x)
