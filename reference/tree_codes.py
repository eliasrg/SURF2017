"""Stack algorithm for decoding convolutional codes.
Written by Anatoly Khina (khina@caltech.edu)
Downloaded from http://www.its.caltech.edu/~khina/code/tree_codes.py
on 2017-07-26 and converted into Python 3."""

import numpy        as np
import numpy.matlib as ml
import scipy        as sp
import bisect       as bisect
import itertools    as it
##import random as rd

def generate_G(k, n, n_branch):
    """
    @k:        Number of input bits (columns) of each branch (sub-) matrix
    @n:        Number of output bits (rows) of each branch (sub-) matrix
    @n_branch: Number of branches
    return:    Block-Toeplitz lower-triangular generating matrix G
    """
    # Generate sub matrices Gi with i.i.d. Ber(1/2) entries
    Gis = [np.random.randint(0, 2, [n, k]) for x in range(n_branch)]

    # Generate G
    G = np.matrix( np.zeros([n * n_branch, k * n_branch], 'int16') )
    for i, g in enumerate(Gis):
        G[i*n:, :(n_branch-i)*k] += sp.kron(sp.eye(n_branch - i, dtype=np.int16), Gis[i])

    return G

def generate_noise(n, n_branch, p):
    """
    @n:        Number of ENC output bits at each stage/branch
    @n_branch: Number of ENC branches
    @p:        Crossover probability of Ber(p) noise
    return:    i.i.d. Ber(p) noise vector of length n * n_branch
    """
    return np.matrix( np.random.binomial(1, p, [n * n_branch, 1]) )

def calc_E0(rho, p):
    """
    Calculates Gallager's E0(rho) for BSC(p).
    E0(1) is the cutoff rate.
    """
    return rho - (1 + rho) * np.log2( p**(1. / (1+rho)) + (1-p)**(1. / (1+rho)) )

def stack_dec(y, G, k, n, n_branch, p, bias = 'R', output_paths = 'best', flag_print_stat = False):
    """
    Runs the stack algorithm. Assumes a BSC(p) and a Fano metric with bias R, E0(1) or prescribed number
    Remark: For a different channel change the line following the line "Calculate metric increment"
    @y: Column output vector of length n * n_branch
    @G: Generating matrix of dimensions (n * n_branch) x (k * n_branch)
    @p: Crossover probability of a BSC(p)
    @bias: 'R' (min complexity) or 'E0' (min error prob.) depending on which bias to use
    @output_paths: Takes one of the following values:
                   'best': Returns first path of length len(y) to reach stack top
                   'all':  Returns all paths that reach stack top during the algo.
                   'lens': For each 1 <= L <= len(y), store first path to reach stack top of length L
    @flag_print_stat: Prints length and metric of current path (@stack top)
    return: (metric, source path, code path) of path that first gets to top of stack of length(y),
            or a list of all paths that reached top of stack during algo.,
            or a list of paths that first reached stack top for each length
    """
    len_y = len(y)
##    n, k = G.shape
    R = float(k) / n
##    n_branch = len_y / n

    # Decide on Bias
    if bias == 'R':
        bias = R
    elif bias == 'E0':
##        bias = 1 - 2 * np.log2(np.sqrt(p) + np.sqrt(1-p))
        bias = calc_E0(1, p)

    # Partition received vector y into columns vectors of length n
    y_branches = y.reshape(len_y // n, n).transpose()

    # Generate all possible k-tuples and assign as columns of a matrix
    bs = np.matrix( list(it.product([0,1], repeat=k)) ).transpose()

    # List of paths. Each path is a tuple of (metric, source_path, code_path)
    paths   = [( 0, np.zeros([0, 1], 'int16'), np.zeros([0, 1], 'int16') )]
    # List of corresponding metrics: metrics = [r[0] for r in paths]
    # Needed because bisect is stupid and cannot handle tuples well
    metrics = [0]

    if output_paths in ('all', 'lens'):
        all_paths = []

    while len(paths[-1][2]) < len_y:
        # Pop path with best metric
        top_path   = paths.pop()
        top_metric = metrics.pop()
        n_branches = len(top_path[1]) // k

        if output_paths in ('all', 'lens'):
            all_paths.append(top_path)

        if flag_print_stat:
            print("length of top  path =", n_branches, "; metric = ", top_path[0])

        # Calculate last branch of x corresponding to current path (excluding last source branch)
        x_last_br  = G[n_branches * n : (n_branches+1) * n, : n_branches * k] * top_path[1]
        x_last_br %= 2

        # Clown path 2**k times = number of branched out paths
        xs_last_br = ml.repmat(x_last_br, 1, 1<<k)

        # Calculate element of last branch of x (code) corresponding to last source branch (k)
        try:
            xs_last_br ^= G[n_branches * n : (n_branches+1) * n, n_branches * k : (n_branches+1) * k] * bs
        except:
            print('--------------- ERROR ----------------')
            print("n_branches = ", n_branches, "n = ", n, "k = ", k)
            print('--------------- ERROR ----------------')
            return G[n_branches * n : (n_branches+1) * n, n_branches * k : (n_branches+1) * k], bs

        # Create and append the new paths
        for i in range(1<<k):
            # Calculate noise vector (given this is the correct path)
            x_last_br = xs_last_br[:,i]
            z_last_br = x_last_br ^ y_branches[:,n_branches]

            # Calculate metric increment
            metric  = np.sum(z_last_br > .5) * np.log2(p) + sp.sum(z_last_br <= .5) * np.log2(1-p) + n

            # Subtract sequential-decoding bias (G in Jelinek's book; B in Gallager's book)
            metric -= n*bias

            # Add to stored metric
            metric += top_metric

            # Concat source branch to source path
            b = np.concatenate([top_path[1], bs[:,i]])

            # Concat code branch to code path
            x = np.concatenate([top_path[2], x_last_br])

            # Add path to list of paths and sort (ascending) according to metric
            bisect_ind = bisect.bisect(metrics, metric)
            metrics.insert(bisect_ind, metric)
            paths.insert(  bisect_ind, (metric, b, x))

    # Return the desired output (depdning on the value of output_paths)
    if output_paths in ("all", "lens"):
        # Add last path (of length len(y)) to all_paths
        all_paths.append(paths[-1])
        if output_paths == "all":
            return all_paths

        # Find first path in all_paths of every length
        return [[x for x in all_paths if len(x[2]) == L][0] for L in range(n, len_y + n, n)]
    else:
        return paths[-1]

