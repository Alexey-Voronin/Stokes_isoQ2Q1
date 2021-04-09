import numpy as np
import scipy
import scipy.sparse as sp
import pyamg
from pyamg.relaxation.relaxation import jacobi, gauss_seidel


class BraessSarazinSmoother:
    """
    Braess-Sarazin relaxation for 2x2 block matrices.

    Braess-Sarazin relaxation that implements different ways of solving the
    Schur complement.

    """

    def __init__(self, stokes, params, leg='down'):
        """
        Braess-Sarazin initialization function.

        Args:
            stokes: sysmg.systems.stokes.StokesSystem
                Stokes object used to pass in Stokes submatrices: A, M , BT, B,
                and Zero_block which does not need to be zero

            params: dict
                Dictionary object specifying all the parameters needed for BS
                relaxation.

                Keys:
                'vcycle'          : tuple
                    (# of pre-, #  of post-) relaxation sweeps.
                'pressure-solver' : {'diag-schur-exact', 'Gauss-Seidel',
                'multigrid', 'Jacobi', 'schur-exact'}
                    Type of Schur complement solver.
                'alpha'           : float
                    BS vector laplacian scaling factor.
                'omega_smooth'    : float
                    Omega for Jacobi pressures-solver.
                'omega_g'         : float
                    global update damping.

                Example:
                    {'vcycle'         :   (1, 1), 'alpha'        : 1,
                     'pressure-solver': 'Jacobi', 'omega_smooth' :  2./3.,
                     'psolve_iters'   :        1, 'omega_g'      : 0.9}

            leg : {'down', 'up'}
                Specifies whether pre- or post- relaxation is being setup.

        Returns:
            Nothing

        """
        self.params                 = params
        self.SchurComplementSolver  = params['pressure-solver']
        self.relax_iters            = params['vcycle'][0] if leg == 'down' else params['vcycle'][1]
        self.psolve_iters           = params['psolve_iters'] if 'psolve_iters' in params.keys() else 1.0
        self.alpha                  = params['alpha']
        self.omega_g                = params['omega_g'] if 'omega_g' in params.keys() else 1.0
        self.omega_smooth           = -1
        if self.SchurComplementSolver == 'Jacobi':
            self.omega_smooth       = params['omega_smooth']

        if self.omega_g  < 1e-8:
            self.relax_iters = 0

        self.setup  = False
        #input
        self.A      = stokes.A
        self.M      = stokes.M
        self.BT     = stokes.BT
        self.B      = stokes.B
        self.Z      = stokes.Zero_block
        self.vlen   = self.M.shape[0]
        self.plen   = self.M.shape[1]
        # Schur complement
        self.S      = None
        self.S_inv  = None
        self.ml     = None # if SchurComplementSolver == 'multigrid'
        # need to construct (once)
        self.Mapprox_inv   = None

        if self.SchurComplementSolver  in ['diag-schur-exact', 'Gauss-Seidel', 'multigrid', 'Jacobi']:
            C         = self.M.diagonal()
            nnz       = C.nonzero()
            Cinv      = np.zeros_like(C)
            Cinv[nnz] = 1./C[nnz]
            self.Mapprox_inv  = sp.spdiags(Cinv, 0, C.size, C.size)
        elif self.SchurComplementSolver == 'schur-exact':
            self.Mapprox_inv  = sp.csr_matrix(np.linalg.pinv(self.M.toarray()))
        else: # ignore
            self.Mapprox_inv = None
            raise ValueError('BraessSarazinSmoother.__init__ FAIL: non-existant SchurComplementSolver=%s' % SchurComplementSolver)

    def relax(self, A, x, b, alpha = None, callback=None):
        """
        Apply Braess-Sarazin relaxation.

        Args:
            A: (csr_matrix)
                Systems matrix (ignored, but needed to be compatible with pyamg relaxation).

            x: (numpy.ndarray)
                Initial guess.

            b: (numpy.ndarray)
                Right hand side.

            alpha: (float)
                Optional parameter to change alpha variable/

            callback: (function)
                Records the residual at each iteration of relax.

        Returns:
            (numpy.ndarray):  Solution vector.

        Notes:
            If you would like to chain parameters then modify the relaxation object via the helper functions.

        """
        vlen       = self.vlen
        # component wise solution
        sol    = x.copy()
        u      = sol[0:vlen]
        p      = sol[vlen:]
        # component wise update
        du     = np.zeros_like(u)
        dp     = np.zeros_like(p)
        # component wise residual
        ru     = np.zeros_like(u)
        rp     = np.zeros_like(p)

        if alpha is not None and abs(alpha-self.alpha) > 1e-4:
            self.alpha = alpha
            self.S = None

        if self.S is None:
            self.S = -1*(self.Z+(1.0/self.alpha)*self.B*self.Mapprox_inv*self.BT)
            self.S = self.S.tocsr()
            if  self.SchurComplementSolver == 'multigrid':
                self.ml = pyamg.smoothed_aggregation_solver(self.S,
                            strength='symmetric',
                            aggregate='standard',
                            smooth=('energy'),
                            presmoother =('gauss_seidel', {'sweep':'symmetric', 'iterations':1}),
                            postsmoother=('gauss_seidel', {'sweep':'symmetric', 'iterations':1}),
                            improve_candidates= None,
                            max_coarse=5,
                            keep=False)

            if  self.S_inv is not None:
                self.S_inv = np.linalg.pinv(self.S.toarray())

        for i in range(0, self.relax_iters):
            # split up residual
            ru[:]   = (b[:vlen]-(self.M*u+self.BT*p))
            rp[:]   = (b[vlen:]-(self.B*u-self.Z*p ))
            # zero out previous update
            du[:] = 0.0
            dp[:] = 0.0

            # compute update: Sy = g - (1/alpha)BC^{-1}d
            rhs = rp - self.B*self.Mapprox_inv*ru/self.alpha
            if self.SchurComplementSolver == 'Gauss-Seidel':
                gauss_seidel(self.S, dp, rhs, iterations=self.psolve_iters, sweep='symmetric')
            elif self.SchurComplementSolver == 'Jacobi':
                jacobi(self.S, dp, rhs, iterations=self.psolve_iters, omega=self.omega_smooth)
            elif self.SchurComplementSolver in ['diag-schur-exact', 'schur-exact']:
                if  self.S_inv is None:
                    self.S_inv = np.linalg.pinv(self.S.toarray())
                dp =self.S_inv@rhs
            else: #multigrid
                dp = self.ml.solve(rhs, maxiter=self.psolve_iters)

            du = (1.0/self.alpha)*self.Mapprox_inv*(ru-self.BT*dp)
            # update solution components
            u[:]   = u + self.omega_g*du
            p[:]   = p + self.omega_g*dp

            if callback is not None:
                callback(b-self.A*sol)

        return sol


    def print_params(self):
        print('self.SchurComplementSolver=%s: (alpha= %.2f, omega_g=%.2f, omega_smooth%.2f), (relax_iters=%d, psolve_iters=%d)' % (self.SchurComplementSolver, self.alpha, self.omega_g,  self.omega_smooth, self.relax_iters, self.psolve_iters))

    def update_alpha(self,alpha):
        # need to rebuild S
        if abs(alpha-self.alpha) > 1e-4:
             self.alpha = alpha
             self.S = None
    def update_omega_g(self,omega_g):
        self.omega_g = omega_g

    def update_omega_smooth(self, omega_smooth):
        self.omega_smooth = omega_smooth

    def update_relax_iters(self, relax_iters):
        self.relax_iters = relax_iters

    def relax_inplace(self, A, x, b):
        """
        Apply inplace Braess-Sarazin relaxation.

        Args:
            A: (csr_matrix)
                Systems matrix (ignored, but needed to be compatible with pyamg
                relaxation).

            x: (numpy.ndarray)
                Initial guess.

            b: (numpy.ndarray)
                Right hand side.

        Returns:
            None, x will be modified in place.

        Notes:
            If you would like to chain parameters then modify the relaxation
            object via the helper functions.

        """
        x[:] = self.relax(A, x, b)
