import numpy                     as np
from scipy.sparse                   import csr_matrix, identity
from scipy.sparse.linalg            import LinearOperator
from pyamg.multilevel               import multilevel_solver

from .relaxation.relaxation_wrapper import RelaxationWrapper
from .relaxation.braesssarazin      import *
from .relaxation.vanka              import *
from .geometric.geometric_interp    import *
from .systems.stokes                import *


class MG:
    """Construct a monolithic multigrid solver for Stokes System."""

    velocity_Ps = None # velocity interpolation operators
    pressure_Ps = None # pressure interpolation operators
    ml_unified  = None # monolithic multigrid solver for stokes system
    stokes_sys  = None # list of stokes_sys for each level of ml_unified hierarchy

    def __init__(self, stokes, LEVELs,
                 input_args = dict(multigrid_type='geometric', interpolation_type=(1,1),
                                   coarsenining_rate=(2,2),
                                   smoother=('Braess-Sarazin', {'vcycle' : (2,2), 'alpha' : 1.,
                                             'pressure-solver': 'Jacobi',
                                             'psolve_iters': 3, 'omega' : 1.5, 'omega_g': 1}))):
        """
        Construct pyamg.multilevel object.

        Construct multigrid solver based on pyamg.multilevel object. This method
        focused on copuled monolithic algebraic and geometric MG for Stokes System.

        Args:
            stokes: sysmg.systems.stokes.StokesSystem object
                Object storing all the relevant Stokes system information.
            LEVELs: int
                Suggested level for the MG hierarchy. In case of GMG, the hierarchy
                depth is usually exactly this number, while for AMG this is an
                upper bound.
            input_args: dict
                Input parameters specifying the type of MG hierarchy.

                Options:
                'multigrid_type'     : {'geometric', 'algebraic'}.
                'interpolation_type' : tuple specifying order of geometric interpolation.
                'coarsenining_rate'  : tuple containing geometric corseing rate in
                                        each dimension.
                'smoother'           : Stokes system relaxation parameters. See
                                        sysmg.relaxation for more details.


        Returns:
            type: description

        Raises:
            Exception: description

        """
        multigrid_type      = input_args['multigrid_type']
        interpolation_type  = input_args['interpolation_type']

        Mc  = BTc = Bc  = Zc  = None
        A   = stokes.A
        M   = stokes.M
        BBT = stokes.B*stokes.B.T
        B   = stokes.B
        BT  = stokes.BT
        Z   = stokes.Zero_block
        Mass= stokes.Mass_p
        pStiff=stokes.Pressure_stiff
        stokes_coamg_next_lvl = None

        if  'smoother' not in input_args.keys():
            input_args['smoother'] = ('Braess-Sarazin', {'vcycle' : (2,2), 'alpha' : 1.2,
                                      'pressure-solver': 'Jacobi',
                                      'psolve_iters': 3, 'omega' : 2./3., 'omega_g': 1})

        if multigrid_type == 'geometric':
            coarsenining_rate   = input_args['coarsenining_rate']
            Ps_all              = geometric_interp(stokes,  coarsenining_rate, interpolation_type)

            velocity_Ps         = Ps_all['Velocity DOFs'][::-1]
            pressure_Ps         = Ps_all['Pressure DOFs'][::-1]
        else:
            raise ValueError('Error: %s not defined' % str(multigrid_type))

        ########################################################################################
        # Construct Multigrid Hierarchy
        ########################################################################################
        Mc  = BTc = Bc  = Zc  = None
        A   = stokes.A
        M   = stokes.M
        BBT = stokes.B*stokes.B.T
        B   = stokes.B
        BT  = stokes.BT
        Z   = stokes.Zero_block
        Mass= stokes.Mass_p
        pStiff=stokes.Pressure_stiff
        stokes_coamg_next_lvl = None

        # store these matrices to construct smoother later
        stokes_lvl             = StokesSystem()
        stokes_lvl.A           = A
        stokes_lvl.M           = M
        stokes_lvl.B           = B
        stokes_lvl.BT          = BT
        stokes_lvl.Zero_block  = Z
        stokes_lvl.Mass_p      = Mass
        stokes_lvl.Pressure_stiff = pStiff
        # needed for co-agg-amg
        stokes_lvl.p_dof_coord_hier = [stokes.p_dof_coord_hier[-1]]
        stokes_lvl.v_dof_coord_hier = [stokes.v_dof_coord_hier[-1]]
        stokes_lvl.NEx_hier         = [stokes.NEx_hier[-1]]
        stokes_lvl.NEy_hier         = [stokes.NEy_hier[-1]]
        stokes_lvl.p_grid_hier      = [stokes.p_grid_hier[-1]]
        stokes_lvl.v_grid_hier      = [stokes.v_grid_hier[-1]]
        stokes_lvl.v_ext_grid_hier  = [stokes.v_ext_grid_hier[-1]]
        stokes_lvl.bcs_v_nodes_hier = [stokes.bcs_v_nodes_hier[-1]]
        stokes_lvl.bcs_p_nodes_hier = [stokes.bcs_p_nodes_hier[-1]]
        stokes_lvl.domain           = stokes.domain
        stokes_lvl.periodic         = stokes.periodic
        stokes_sys                  = [ stokes_lvl ]

        levels = []
        levels.append(multilevel_solver.level())
        levels[-1].A = A

        actual_lvls = 1
        for lvl in range(LEVELs-1):
            ############################################################
            # Set Matrices
            if lvl > 0:
                A   = stokes_sys[-1].A
                M   = Mc
                BBT = Bc*BTc
                B   = Bc
                BT  = BTc
                Z   = Zc
                Mass=Massc
                pStiff=pStiffc

            if M.shape[0] < 100:
                break # stop coarsening
            else:
                actual_lvls += 1

            ############################################################
            # Interpolation
            Pv = None; Pp = None
            Rv = None; Rp = None
            if multigrid_type == 'geometric':

                Pvx = csr_matrix(velocity_Ps[lvl])

                Rv =  sp.bmat([[Pvx,   None],
                               [None,   Pvx]]).tocsr().T
                Rp  = csr_matrix(pressure_Ps[lvl]).T

                Pv  = Rv.copy().T*0.25
                Pp  = Rp.copy().T*0.25
                # Assemble P
                levels[-1].P   = sp.bmat([[Pv,   None],
                                          [None,   Pp]]).tocsr()
                levels[-1].R   = sp.bmat([[Rv,   None],
                                           [None,   Rp]]).tocsr()

            else:
                raise ValueError('Error: %s not defined' % str(interpolation[0]))

            levels.append(multilevel_solver.level())

            ############################################################
            # Save matrices for relaxation set-up
            Mc                  = Rv*M*Pv
            BTc                 = Rv*BT*Pp
            Bc                  = Rp*B*Pv
            Zc                  = Rp*Z*Pp
            Massc               = Rp*Mass*Pp
            pStiffc             = Rp*pStiff*Pp
            Ac                  = sp.bmat([[Mc, Bc.T], [Bc, Zc]])
            levels[-1].A        = Ac

            stokes_lvl          = StokesSystem()
            stokes_lvl.A        = Ac
            stokes_lvl.M        = Mc
            stokes_lvl.B        = Bc
            stokes_lvl.BT       = BTc
            stokes_lvl.Zero_block = Zc
            stokes_lvl.Mass_p   = Massc
            stokes_lvl.pStiff   = pStiffc
            stokes_lvl.NEx_hier = [stokes_sys[-1].NEx_hier[-1]//2]
            stokes_lvl.NEy_hier = [stokes_sys[-1].NEy_hier[-1]//2]
            stokes_lvl.periodic = stokes_sys[-1].periodic
            stokes_lvl.domain           = stokes.domain

            if multigrid_type == 'geometric':
                stokes_lvl.p_dof_coord_hier = [stokes.p_dof_coord_hier[::-1][1+lvl]]
                stokes_lvl.v_dof_coord_hier = [stokes.v_dof_coord_hier[::-1][1+lvl]]
                stokes_lvl.p_grid_hier      = [stokes.p_grid_hier[::-1][1+lvl]]
                stokes_lvl.v_grid_hier      = [stokes.v_grid_hier[::-1][1 + lvl]]
                stokes_lvl.v_ext_grid_hier  = [stokes.v_ext_grid_hier[::-1][1 + lvl]]
                stokes_lvl.bcs_v_nodes_hier = [stokes.bcs_v_nodes_hier[::-1][1+lvl]]
                stokes_lvl.bcs_p_nodes_hier = [stokes.bcs_p_nodes_hier[::-1][1+lvl]]

            stokes_sys.append(stokes_lvl)

        ############################################################################
        # Setup Relaxation
        ml_unified = multilevel_solver(levels)
        smoother   = input_args['smoother']
        for lvl in range(actual_lvls-1):
            stokes_lvl              = stokes_sys[lvl]

            if multigrid_type == 'geometric':
                field                   = 'Velocity DOFs'
                stokes_lvl.v_dof_coord  = stokes.v_dof_coord_hier[::-1][lvl]
                stokes_lvl.v_grid       = stokes.v_grid_hier[::-1][lvl]

                field                   = 'Pressure DOFs'
                stokes_lvl.p_dof_coord  = stokes.p_dof_coord_hier[::-1][lvl]
                stokes_lvl.p_grid       = stokes.p_grid_hier[::-1][lvl]
                stokes_lvl.periodic     = stokes.periodic

            if  smoother[0] == 'Braess-Sarazin':
                if lvl > 0 and smoother[1]['omega_g'] < 1e-8:
                    input_args['smoother'][1]['omega_g'] = input_args['smoother'][1]['omega_g2']
                    input_args['smoother'][1]['alpha']   = input_args['smoother'][1]['alpha2']
                    input_args['smoother'][1]['omega']   = input_args['smoother'][1]['omega2']
                bs_pre  = BraessSarazinSmoother(stokes_lvl, params=smoother[1], leg="down")
                bs_post = BraessSarazinSmoother(stokes_lvl, params=smoother[1], leg='up')
            elif smoother[0] == 'Vanka':
                tmp = input_args['debug'] if 'debug' in input_args.keys() else False
                bs_pre  = Vanka(stokes_lvl, params=smoother[1], leg="down")
                bs_post = Vanka(stokes_lvl, params=smoother[1], leg='up', Vanka_given=bs_pre)
            else:
                raise Exception("Smoother name is wrong")

            post_relax = RelaxationWrapper(bs_post, bs_post.relax_inplace)
            pre_relax  = RelaxationWrapper(bs_pre , bs_pre.relax_inplace)

            ml_unified.levels[lvl].presmoother   = pre_relax
            ml_unified.levels[lvl].postsmoother  = post_relax
        ############################################################################
        # coarse grid solver
        ml_unified.coarse_solver = pyamg.multilevel.coarse_grid_solver("pinv2")

        self.velocity_Ps = velocity_Ps
        self.pressure_Ps = pressure_Ps
        self.ml_unified  = ml_unified
        self.stokes_sys  = stokes_sys

def make_precond(stokes, ml):
    ML   = ml.aspreconditioner(cycle='V')
    null = stokes.nullspace

    def mv(v):
        vout = v
        vout = ML*vout
        vout = vout-null*np.dot(null, vout)
        return vout
    return LinearOperator((stokes.A.shape[0],stokes.A.shape[1]), matvec=mv)
