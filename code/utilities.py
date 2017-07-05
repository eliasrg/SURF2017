def memoized(f):
    """Returns a version of f that remembers which arguments it has been called
    with and stores them in a table to avoid recomputing values."""
    table = dict()

    def memo_f(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]

    return memo_f
