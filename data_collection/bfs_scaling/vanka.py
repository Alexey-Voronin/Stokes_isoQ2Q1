import time
import sys
sys.path.append('../../')
from sysmg import *
from sysmg.util.data_analysis import conv_factor, collect_residuals

#####################################
# Solver Params
GMRES        = {'module': ("pyamg", 'fgmres'), "resid": 'abs'}
TOL          = 1e-10
MAXITER      = 100
BC           = 'bfs'
########################################
# system
NEx_all  = [8, 16, 32, 64, 128, 256]
LVLs_all = [2,  3,  4,  5,   6, 7]

stokes_problems = dict()
print('------------------------------------------------')
print('System set-up')
for NEx in NEx_all:
    tic        = time.perf_counter()
    stokes_problems[NEx] = form_stokes(Nelem=(NEx,NEx), domain= (1,1), lo_fe_precond= True,
                                      Plot= False,     Solve= True, quadrilateral= True, BCs=BC)
    toc        = time.perf_counter()
    print('NEx=%d\ttime=%.2f s' % (NEx, toc-tic))

print('\n\nCollect Data:\n')
#####################################
for params in [
                (0.99, 0.67, 0.93, (1,1,1)), # W cycle iterations grow
                (0.80, 0.85, 0.00, (2,2,1)),
                (1.05, 0.74, 0.78, (1,1,2)),
                (0.98, 0.71, 1.05, (2,2,1)),
                ]:
    print('params=', str(params))
    for NEx, lvls in zip(NEx_all, LVLs_all):
        stokes_ho    = stokes_problems[NEx]
        stokes_lo    = stokes_ho.lo_fe_sys
        # only need to update the entire solver when system changes
        gmg_lo_obj   = None
        first_time   = True

        if type(params) == str:
            print(params)
            continue

        tau, omega_0, omega_1, (nu_1, nu_2, gamma)   = params
        smoother_inner  = ('Vanka', {'type'  : 'geometric', 'update': 'additive',
                                     'vcycle':  (1, 1),     'omega_g' : omega_1})
        inner_cycle     =  dict(multigrid_type    = 'geometric', interpolation_type = (1,1),
                                coarsenining_rate = (2, 2),               Q2_iso_Q1 = True,
                                tau      = tau,
                                gamma    = gamma,
                                smoother = smoother_inner)
        outer_cycle     = ('Vanka', {'type'  : 'geometric',  'update' : 'additive',
                                     'vcycle':  (nu_1, nu_2), 'omega_g' : omega_0})

        print('\tNEx=%d\tDOFs=%d\tlvls=%d' % (NEx, stokes_lo.A.shape[0], lvls))
        for cycle_type in ['V', 'W']:
            # assemble/change mnultigrid parameters
            if first_time:
                gmg_lo_obj = iso_mg((stokes_lo, stokes_ho), lvls, inner_cycle, outer_cycle, cycle_type=cycle_type)
                first_time = False
            else:
                gmg_lo_obj.update_params(inner_cycle, outer_cycle, cycle_type=cycle_type)

            gmg_lo     = gmg_lo_obj.get_solver()

            tic        = time.perf_counter()
            x, resid_lo= solve(stokes_ho.A, stokes_ho.b, x0=None, M=gmg_lo,
                                 maxiter=MAXITER, tol=TOL, GMRES=GMRES)
            itc        = len(resid_lo)
            toc        = time.perf_counter()

            print('\t\t\t\t\t\tm_%s=%d\ttime=%.3f' % (cycle_type, itc, toc-tic))
