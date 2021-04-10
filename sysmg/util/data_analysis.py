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


def conv_factor(resids, multi=True, model='lsq', nresid=10):
    """Process residual histories and output mean convergence rate and number of iterations."""
    asym_num    = 10  # last asym_num number of resids
    resids      = resids if multi else [resids]
    cf          = None

    if model == 'geometric':
        cf      = []
        for resid0 in resids:
            resid = resid0[-1 * nresid:]
            r = resid[1:]/resid[:-1]
            cf.append(r.prod()**(1.0/len(r)))
        cf = np.mean(cf)
    elif model == 'lsq':
        cf = lstsq_fit(resids, asym_num=nresid)

    iters       = [len(r) for r in resids]

    return cf, int(np.mean(iters))


def collect_residuals(fsolver, stokes, max_runs=100):
    """Collect residual histories for zero rhs and max_runs number of random intial guesses."""
    seeds = [ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541]
    seeds = seeds[:min(max_runs,100)]

    resids = []
    for r in seeds:
        np.random.seed(r)
        b  = np.zeros(stokes.A.shape[0])
        x0 = np.random.rand(stokes.A.shape[0])
        x0 = x0/np.linalg.norm(x0)

        resids.append(fsolver(b, x0))

    return resids
