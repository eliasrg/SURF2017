from numpy import sqrt
from scipy.interpolate import interp1d
import scipy.stats as stats

def gaussian(var):
    """Returns a Gaussian PDF with variance var."""
    stddev = sqrt(var)
    return stats.norm(0, stddev)


class Custom(stats.rv_continuous):
    def __init__(self, x, fx):
        a = x[0]
        b = x[-1]
        stats.rv_continuous.__init__(self, a=a, b=b)

        self.x = x
        self.fx = x
        self._pdf = interp1d(x, fx,
                kind='linear', bounds_error=False, fill_value=0)
