import numpy as np
import scipy.stats.distributions as ssd

import moresque.ustructures as us

def main(val, t):
    mean = val
    std = 0.2 * val

    dstr = ssd.norm(mean, std)
    vals = np.linspace(dstr.ppf(1e-5), dstr.ppf(1 - 1e-5), 1000)
    weis = dstr.pdf(vals)

    return us.Distribution(values=vals, weights=weis, num=1000)
