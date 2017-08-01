import numpy as np
from numpy.linalg import matrix_rank

def memoized(f):
    """Returns a version of f that remembers which arguments it has been called
    with and stores them in a table to avoid recomputing values."""
    table = dict()

    def memo_f(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]

    return memo_f

def int_binsearch(p, lo, hi):
    """Takes a predicate p: int -> bool.
    Assumes that there is some I (lo <= I < hi) such that
    p(i) = (i >= I). Finds and returns I, or None if no such I exists."""
    if lo == hi:
        return None
    elif lo + 1 == hi:
        return lo if p(lo) else None
    else:
        mid = (lo + hi - 1) // 2
        if p(mid): # mid >= I
            return int_binsearch(p, lo, mid + 1)
        else: # mid < I
            return int_binsearch(p, mid + 1, hi)


# Example: Search in list
#
# p = lambda i: return X[i]
#
# I = int_binsearch(p, 0, len(X));
# x = X[I];


def to_column_vector(x):
    return np.reshape(x, [-1, 1])

def full_rank(M):
    return matrix_rank(M) == min(M.shape)
