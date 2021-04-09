from scipy.sparse.linalg            import LinearOperator
from .h_multigrid                   import MG
from .relaxation.braesssarazin      import *
from .relaxation.vanka              import *
from .relaxation.relaxation_wrapper import RelaxationWrapper

class iso_mg(object):
    """
    Construct ph-multigrid solver for Stokes system.

    Currently this ph-multigrid solver only works for Q2/Q1 discretization,
    and utilizes Q1isoQ2/Q1 discretization to construcft h-multigrid solver.

    """

    # systems
    stokes_inner = None
    stokes_outer = None
    # parameters
    inner_relax  = None
    outer_relax  = None
    cycle_type   = None
    tau          = None
    # Objects to be updated
    outer_prerelax= None
    outer_postrelax= None
    gamma        = None
    h_mg_object  = None
    h_mg        = None
    solver_lin_op= None

    def __init__(self, stokes, LEVELs, inner_cycle, outer_cycle, cycle_type='V'):
        """
        Construct monolithic ph-multigrid object for Q2/Q1 system.

        This method performs pre- and post- relaxation on the fine level of Q2/Q1
        system, performs standard h-multigrid on the companion Q1isoQ2/Q1 discretization
        of the same problem.

        Args:
            stokes: sysmg.systems.stokes.StokesSystem object
                Object storing all the relevant Stokes system information.
            LEVELs: int
                Suggested level for the MG hierarchy. In case of GMG, the hierarchy
                depth is usually exactly this number, while for AMG this is an
                upper bound.
            inner_cycle: dict
                Parameters for the Q1isoQ2/Q1 h-multigrid.

                Example:
                    {'multigrid_type'    = 'geometric',
                    'interpolation_type' = (1,1),
                    'coarsenining_rate'  = (2, 2),
                    'Q2_iso_Q1'          = True,
                              'tau'      = 1,
                              'gamma'    = 1,
                              'smoother' = smoother_inner}
                    where
                    smoother_inner  = ('Vanka', {'type'   : 'geometric',
                                                 'update' : 'additive',
                                                 'vcycle' :  (1, 1),
                                                 'omega_g': 1})
            outer_cycle: dict
                Parameters for the relaxation on the fine level of the Q2/Q1 system.

                Examples:
                        outer_cycle     = ('Vanka', {'type'   : 'geometric',
                                                     'update' : 'additive',
                                                      'vcycle': (1, 1),
                                                     'omega_g': 1})

        """
        self.inner_relax = inner_cycle['smoother']
        self.tau         = inner_cycle['tau'] #if "tau" in inner_cycle.keys() else 1.0
        self.gamma       = inner_cycle['gamma'] # if "gamma" in inner_cycle.keys() else 1
        self.outer_relax = outer_cycle
        self.cycle_type  = cycle_type

        if type(stokes) is not tuple:
            self.stokes_inner = stokes
            self.stokes_outer = stokes
        elif len(stokes) == 2:
            self.stokes_inner = stokes[0]
            self.stokes_outer = stokes[1]
        else:
            raise ValueError("Input Stokes system or tuple of Stokes systems (stokes_inner, stokes_outer)")

        self.h_mg_object    = MG(self.stokes_inner, LEVELs, input_args=inner_cycle)
        self.h_mg          = self.h_mg_object.ml_unified
#        (self.velocity_Ps, self.pressure_Ps),  (self.ml_vx, self.ml_p),  self.h_mg, self.stokes_sys = bla

        if outer_cycle[0] == 'Braess-Sarazin':
            self.outer_prerelax  = BraessSarazinSmoother(self.stokes_outer, params=self.outer_relax[1], leg="down")
            self.outer_postrelax = BraessSarazinSmoother(self.stokes_outer, params=self.outer_relax[1], leg='up')
        elif outer_cycle[0] == 'Vanka':
            self.outer_prerelax  = Vanka(self.stokes_outer, params=self.outer_relax[1], leg="down")
            self.outer_postrelax = Vanka(self.stokes_outer, params=self.outer_relax[1], leg='up')
        else:
            raise ValueError("Incorrect outer_cycle relaxation")

        # assembly happens outside of init to facilicate change of Parameters
        # without having to rebuild MG hierarchy and relaxation objects
        self.assemble_solver()


    def assemble_solver(self):
        """Assemble ph-multigrid system based on the saved parameters."""
        shape  = self.stokes_inner.A.shape
        dtype  = self.stokes_inner.A.dtype

        ones = np.ones((self.stokes_outer.A.shape[0],))
        ones_norm = np.linalg.norm(ones)
        def matvec(b):
            x = np.zeros_like(b)
            self.outer_prerelax.relax_inplace(self.stokes_outer.A, x, b)

            x[:] += self.tau * self.h_mg.solve(b-self.stokes_outer.A*x, x0=None,
                                                maxiter=self.gamma, cycle=self.cycle_type)
            self.outer_postrelax.relax_inplace(self.stokes_outer.A, x, b)
            if self.stokes_outer.periodic:
                x -= np.dot(x, ones) * ones / ones_norm
            return x

        self.solver_lin_op = LinearOperator(shape, matvec, dtype=dtype)


    def update_params(self, inner_cycle, outer_cycle, cycle_type='V'):
        """Update ph-multigrid parameter."""
        self.cycle_type       = cycle_type
        self.inner_relax      = inner_cycle['smoother']
        self.tau              = inner_cycle['tau']   #if "tau" in inner_cycle.keys() else 1.0
        self.gamma            = inner_cycle['gamma'] #if "gamma" in inner_cycle.keys() else 1
        self.outer_relax      = outer_cycle

        if self.outer_relax[0] == 'Braess-Sarazin':
            self.outer_prerelax.update_omega_smooth(self.outer_relax[1]['omega_smooth'])
            self.outer_postrelax.update_omega_smooth(self.outer_relax[1]['omega_smooth'])
            self.outer_prerelax.update_alpha(self.outer_relax[1]['alpha'])
            self.outer_postrelax.update_alpha(self.outer_relax[1]['alpha'])
        elif self.outer_relax[0] == 'Vanka':
            bla = None
        else:
            raise ValueError("Incorrect outer_cycle relaxation")
        self.outer_prerelax.update_omega_g(self.outer_relax[1]['omega_g'])
        self.outer_postrelax.update_omega_g(self.outer_relax[1]['omega_g'])
        self.outer_prerelax.relax_iters  = self.outer_relax[1]['vcycle'][0]
        self.outer_postrelax.relax_iters = self.outer_relax[1]['vcycle'][1]

        for lvl in range(len(self.h_mg.levels)-1):
            if self.inner_relax[0] == 'Braess-Sarazin':
                self.h_mg.levels[lvl].presmoother.relax_obj.update_omega_smooth(self.inner_relax[1]['omega_smooth'])
                self.h_mg.levels[lvl].postsmoother.relax_obj.update_omega_smooth(self.inner_relax[1]['omega_smooth'])

                self.h_mg.levels[lvl].presmoother.relax_obj.update_alpha(self.inner_relax[1]['alpha'])
                self.h_mg.levels[lvl].postsmoother.relax_obj.update_alpha(self.inner_relax[1]['alpha'])
            elif self.inner_relax[0] == 'Vanka':
                bla = None
            else:
                raise ValueError("Incorrect outer_cycle relaxation")
            self.h_mg.levels[lvl].presmoother.relax_obj.update_omega_g(self.inner_relax[1]['omega_g'])
            self.h_mg.levels[lvl].postsmoother.relax_obj.update_omega_g(self.inner_relax[1]['omega_g'])
            self.h_mg.levels[lvl].presmoother.relax_obj.update_relax_iters(self.inner_relax[1]['vcycle'][0])
            self.h_mg.levels[lvl].postsmoother.relax_obj.update_relax_iters(self.inner_relax[1]['vcycle'][1])

        self.assemble_solver()

    def get_solver(self):
        """Return construct ph-multigrid solver as a linear operator."""
        return self.solver_lin_op
