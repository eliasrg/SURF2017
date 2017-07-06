import scipy.stats as st
from math import sqrt

def gaussian(var):
    """Returns a Gaussian PDF with variance var."""
    stddev = sqrt(var)
    return st.norm(0, stddev)
