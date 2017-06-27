def memoized(f):
    table = dict()

    def memo_f(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]

    return memo_f
