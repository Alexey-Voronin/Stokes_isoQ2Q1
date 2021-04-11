import sys
sys.path.append('../../')
from sysmg import *
from sysmg.util.data_analysis import conv_factor, collect_residuals
import time

def collect(relaxation, NExs=[32], problems=['bfs'],
           solver_type={'module': "stationary",        "resid": 'abs'},
           cycle_types=['V', 'W'], mg_lvls=[4,  5, 6, 7],
           tol=1e-10, maxiter=100):

    for BC in problems:
        stokes_problems = dict()
        print('------------------------------------------------')
        print('System set-up: ', BC)
        for NEx in NExs:
            tic        = time.perf_counter()
            stokes_problems[NEx] = form_stokes(Nelem=(NEx,NEx), domain= (1,1), lo_fe_precond= True,
                                               Plot= False,     Solve= True, quadrilateral= True, BCs=BC)
            toc        = time.perf_counter()
            print('NEx=%d\ttime=%.2f s' % (NEx, toc-tic))

        relaxation_type, gmg_params = relaxation
        #####################################
        for params in gmg_params:
            print(params)
            for NEx, lvls in zip(NExs, mg_lvls):
                stokes_ho    = stokes_problems[NEx]
                stokes_lo    = stokes_ho.lo_fe_sys
                # only need to update the entire solver when system changes
                gmg_lo_obj   = None
                first_time   = True

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
                    smoother_inner  = ('Vanka', {'type'  : 'geometric', 'update': 'additive-opt',
                                                 'vcycle':  (1, 1),     'omega_g' : omega_1})
                    inner_cycle     =  dict(multigrid_type    = 'geometric', interpolation_type = (1,1),
                                            coarsenining_rate = (2, 2),               Q2_iso_Q1 = True,
                                             tau      = tau,
                                             gamma    = gamma,
                                             smoother = smoother_inner)
                    outer_cycle     = ('Vanka', {'type'  : 'geometric',  'update' : 'additive-opt',
                                                  'vcycle':  (nu_1, nu_2), 'omega_g' : omega_0})


                print('\tNEx={0:3d}   DoFs={1:8d}   mg_lvls={2:2d}'.format(NEx, stokes_ho.A.shape[0], lvls))
                for cycle_type in cycle_types:
                    # assemble/change mnultigrid parameters
                    if first_time:
                        gmg_lo_obj = iso_mg((stokes_lo, stokes_ho), lvls, inner_cycle, outer_cycle, cycle_type=cycle_type)
                        first_time = False
                    else:
                        gmg_lo_obj.update_params(inner_cycle, outer_cycle, cycle_type=cycle_type)

                    gmg_lo     = gmg_lo_obj.get_solver()
                    times = []
                    for i in range(5):
                        tic        = time.perf_counter()

                        np.random.seed(5)
                        x0 = np.random.rand(stokes_ho.A.shape[0])
                        x, resid_lo= solve(stokes_ho.A, np.zeros_like(x0), x0=x0, M=gmg_lo,
                                              maxiter=maxiter, tol=tol, GMRES=solver_type)
                        itc        = len(resid_lo)
                        toc        = time.perf_counter()
                        times.append(toc-tic)


                    print('\t\t\t\t\t\t{0:1s}   iters={1:3d}   time(s)={2:.3f}'.format(cycle_type, itc, np.mean(times)))

