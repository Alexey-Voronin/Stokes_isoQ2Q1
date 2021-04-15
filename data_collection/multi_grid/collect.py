import sys
sys.path.append('../../')
from sysmg import *
from sysmg.util.data_analysis import conv_factor, collect_residuals


def collect(relaxation, NEx=64, problems=['periodic', 'Washer'],
           solver_types=[{'module': "stationary", "resid": 'abs'}],
           cycle_types=['V', 'W'], mg_lvls=2, tol=1e-10, maxiter=100):

    print('------------------------------------------------')
    print('System set-up')
    stokes_problems = dict()
    for BC in problems:
        print('-%s' % BC)
        stokes_problems[BC] = form_stokes(Nelem=(NEx,NEx), domain= (1,1), lo_fe_precond= True,
                                          Solve= False, quadrilateral= True, BCs=BC)

    print('\n\nCollect Data:\n')
    for BC in stokes_problems.keys():
        print('BC_type=%s--------------------------------------' % BC)
        stokes_ho    = stokes_problems[BC]
        stokes_lo    = stokes_ho.lo_fe_sys
        # only need to update the entire solver when system changes
        gmg_lo_obj   = None
        first_time   = True

        relaxation_type, gmg_params = relaxation
        #####################################
        for params in gmg_params:
            if type(params) == str:
                print(params)
                continue

            if relaxation_type == 'IBSR':
                (nu_1, nu_2, gamma), tau, omega_0, omega_1, alpha_0, beta_0, alpha_1, beta_1 = params
                smoother_inner=('Braess-Sarazin', {'vcycle'         :   (1, 1), 'alpha'        : alpha_1,
                                                   'pressure-solver': 'Jacobi', 'omega_smooth' :  beta_1,
                                                   'psolve_iters'   :        1, 'omega_g'      : omega_1})
                inner_cycle  =  dict(multigrid_type='geometric', interpolation_type=(1,1),
                                    coarsenining_rate=(2, 2), Q2_iso_Q1=True,
                                    tau     = tau,
                                    gamma   = gamma,
                                    smoother= smoother_inner)
                outer_cycle  = ('Braess-Sarazin', {'vcycle'        : (nu_1, nu_2), 'alpha'        : alpha_0,
                                                  'pressure-solver':     'Jacobi', 'omega_smooth' :  beta_0,
                                                  'psolve_iters'   :            1, 'omega_g'      : omega_0})
            else: # Vanka
                (nu_1, nu_2, gamma), tau, omega_0, omega_1   = params
                smoother_inner  = ('Vanka', {'type'  : 'geometric', 'update': 'additive',
                                             'vcycle':  (1, 1),     'omega_g' : omega_1})
                inner_cycle     =  dict(multigrid_type    = 'geometric', interpolation_type = (1,1),
                                        coarsenining_rate = (2, 2),               Q2_iso_Q1 = True,
                                        tau      = tau,
                                        gamma    = gamma,
                                        smoother = smoother_inner)
                outer_cycle     = ('Vanka', {'type'  : 'geometric',  'update' : 'additive',
                                             'vcycle':  (nu_1, nu_2), 'omega_g' : omega_0})


            for cycle_type in cycle_types:
                # assemble/change mnultigrid parameters
                if first_time:
                    gmg_lo_obj = iso_mg((stokes_lo, stokes_ho), mg_lvls, inner_cycle, outer_cycle, cycle_type=cycle_type)
                    first_time = False
                else:
                    gmg_lo_obj.update_params(inner_cycle, outer_cycle, cycle_type=cycle_type)

                gmg_lo     = gmg_lo_obj.get_solver()


                cf_lsq = itc = cf_geom = itc_gmres = None
                for sparam in solver_types:
                    def fsolver(b, x0):
                        x, resid_lo  = solve(stokes_ho.A, b, x0=x0, M=gmg_lo,
                                             maxiter=maxiter, tol=tol, GMRES=sparam)
                        return resid_lo

                    resid_lo    = collect_residuals(fsolver, stokes_ho, max_runs=100)

                    if sparam['module'] == "stationary":
                        cf_lsq, itc = conv_factor(resid_lo, model='lsq')
                        cf_geom, _  = conv_factor(resid_lo, model='geometric')
                    else:
                        _, itc_gmres = conv_factor(resid_lo)

                if itc_gmres == None:
                    print('\t\t{0:1s} {1:55s}  rho_lsq={2:.4f}   rho_geom={3:.4f}   iters={4:3d}'.format(cycle_type, str(params), cf_lsq,cf_geom, itc))
                else:
                    print('\t\t{0:1s} {1:55s}  rho_lsq={2:.4f}   rho_geom={3:.4f}   iters_gmg={4:3d}   iters_gmres={5:3d}'.format(cycle_type, str(params), cf_lsq,cf_geom, itc, itc_gmres))

