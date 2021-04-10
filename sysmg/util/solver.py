import numpy as np
from scipy.sparse.linalg import gmres  as scipy_gmres
from pyamg.krylov        import gmres  as pyamg_gmres
from pyamg.krylov        import fgmres as pyamg_fgmres
from .callback import CallBack_r, CallBack_x

def norm2(x):
    """l2 norm"""
    return np.linalg.norm(x)

# Options:
# maxiter = total number of inner iterations
# GMRES
## modules: pyamg, scipy
## precond: abs, rel
def solve(A, b, x0=None, M=None, maxiter=40, restart=20, tol=1e-8,
          GMRES={'module': ('pyamg', 'gmres'), "resid": 'rel'}, plot=False):
    """
    Linear system iterative solver.

    Linear solver that uses (preconditioned) GMRES or stationary iteration provided
    as linear operator to compute solution to the input linear system.

    Args:
        A: array, matrix, sparse matrix, LinearOperator
            n x n, linear system to solve.
        b: array
            right hand side, shape is (n,) or (n,1)
        x0 : array
            initial guess, default is a vector of zeros.
        M : array, matrix, sparse matrix, LinearOperator
            n x n, inverted preconditioner, i.e. solve M A x = M b.
        maxiter : int
            - default to 40
            - Maximum number of stationary iterations if Krylov method is not used.
            - If Krylov method used, it is passed to the Krylov solver as maxiter.
        restrt: int
            - max number of GMRES inner iterations and maxiter is the max number of
            outer iterations.
        tol: float
            relative convergence tolerance.
        GMRES: dict
            Dictionary containing the information about what type of solver to use
            and what type of residual to report (rel. or abs.).

            Options:
                - 'module' :  'stationary', 'scipy', ('pyamg', 'gmres'), ('pyamg', 'fgmres')
                - "resid"  : 'rel', 'abs'

            Examples:
                - {'module': "stationary",            "resid": 'abs'}
                - {'module': ("pyamg", 'fgmres'), "resid": 'abs'}

    Returns:
        solution vector, and array with residual histoery.

    """

    precond_type = None; module_type = None
    if GMRES is not None:
        precond_type = GMRES["resid"]  if "resid"  in GMRES.keys() else 'rel'
        module_type  = GMRES["module"]  if "module" in GMRES.keys() else 'pyamg'
    else:
        precond_type = 'abs'

    resid   = None
    bnorm   = norm2(b)
    tol     = tol/bnorm if bnorm > 0 else tol
    x0      = np.zeros_like(b) if x0 is None else x0
    cb      = CallBack_x(A, b, M) if precond_type == 'rel' else CallBack_x(A, b)

    x1 = None
    if module_type == 'stationary':
        #print('V-cycle')
        r     = b-A*x0
        x1    = x0
        resid = []
        for i in range(maxiter):
            x1 += M*r
            cb(x1)
            if cb.resid[-1] < tol or (cb.resid[-1]  > 1e8 and i > 50):
                break

            r   = b-A*x1

    elif  module_type == 'scipy':
        #print('scipy-gmres')
        #print('Warning: SciPy GMRES reports residual only at outer iteration')
        cb          = CallBack_r()
        x1, info    = scipy_gmres(A, b, x0=x0, tol=tol, maxiter=maxiter,
                                  M=M, callback=cb,
                                  #callback_type='x', # gives absolute resid but only @outer iter
                                  restart=restart)
    elif module_type == ('pyamg', 'gmres'):
        #print('pyamg-gmres')
        x1, info    = pyamg_gmres(A, b, x0=x0, tol=tol,
                                  restrt=restart, # inner iteration
                                  maxiter=int(maxiter/restart), # outer iter
                                  M=M, callback=cb, orthog='mgs')
    elif module_type == ('pyamg', 'fgmres'):
        #print('pyamg-fgmres')

        x1, info    = pyamg_fgmres(A, b, x0=x0, tol=tol,
                                  restrt=restart, # inner iteration
                                  maxiter=int(maxiter/restart), # outer iter
                                  M=M, callback=cb)
    else:
        raise ValueError('module type is wrong: ' + str(module_type))

    resid       = cb.resid #if "resid" in GMRES.keys() else None
    if bnorm > 1e-8 and module_type != 'scipy':
        resid /= bnorm

    return x1, np.array(resid)
