import numpy as np
import matplotlib.pyplot      as plt

def lstsq_fit(resid_lo, asym_num=10):
    """Least square fir last asym_num numbers and return the slope of the line."""
    X = []
    b = []
    for rin in resid_lo:
        r = np.log(rin[-1 * asym_num:])
        xi = np.linspace(0, len(r) - 1, len(r))

        X += list(xi)
        b += list(r)

    X       = np.array(X)
    b       = np.array(b)
    A       = np.ones((len(b), 2))
    A[:, 0] = X
    par     = np.linalg.lstsq(A, b, rcond=None)[0]

    return np.e ** (par[0])


def conv_factor(resids, multi=False):
    """Process residual histories and output mean convergence rate and number of iterations."""
    asym_num    = 10  # last asym_num number of resids
    resids      = resids if multi else [resids]
    cf_lstsq    = []
    cf_lstsq.append(lstsq_fit(resids, asym_num=asym_num))
    iters       = [len(r) for r in resids]


    return np.mean(cf_lstsq), np.mean(iters)


def collect_residuals(fsolver, stokes, plot=False, max_runs=55):
    """Collect residual histories for zero rhs and max_runs number of random intial guesses."""
    seeds = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
               61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
               127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
               181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
               241, 251, 257, 263, 269, 271, 277]
    seeds = seeds[:min(max_runs,55)]

    resids = []
    b      = np.zeros_like(stokes.A.shape[0])
    for r in seeds:
        np.random.seed(r)
        x0 = np.random.rand(stokes.A.shape[0])
        resids.append(fsolver(b, x0))

    if plot:
        for r in resids:
            plt.semilogy(r)
        plt.show()
    return resids
