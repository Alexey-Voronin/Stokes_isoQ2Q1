import sys
sys.path.append('../../../')
from sysmg import *
from sysmg.util.data_analysis import conv_factor, collect_residuals

#####################################
# Solver Params
GMRES        = {'module': "vcycle", "resid": 'abs'}
TOL          = 1e-10
MAXITER      = 100
mg_lvls      = 5
NEx          = 64

stokes_problems = dict()
print('------------------------------------------------')
print('System set-up')
for BC in ['periodic', 'Washer' ]:
    print('-%s' % BC)
    stokes_problems[BC] = form_stokes(Nelem=(NEx,NEx), domain= (1,1), lo_fe_precond= True,
                                      Solve= False, quadrilateral= True, BCs=BC)

print('\n\nCollect Data:\n')
for BC in ['periodic', 'Washer']:
    print('BC=%s------------------------------------------' % BC)
    stokes_ho    = stokes_problems[BC]
    stokes_lo    = stokes_ho.lo_fe_sys
    # only need to update the entire solver when system changes
    gmg_lo_obj   = None
    first_time   = True

    #####################################
    for params in ["Table 2:",
                   "\t(no post-smoothing)",
#                   (0.86, 0.78, 1.01, (1,0,1)),
                   (0.90, 0.98, 0.83, (1,0,2)),
                   (0.85, 1.02, 0.82, (1,0,3)),
                   "\t(symmetrical)",
                   (0.99, 0.67, 0.93, (1,1,1)),
                   (1.05, 0.74, 0.78, (1,1,2)),
#                   "Table 3:",
#                   "\t(omega_0=0 (no relax on Q2Q1))",
#                   (0.64, 0.00, 0.98, (0,0,1)),
#                   (0.66, 0.00, 0.63, (0,0,2)),
#                   "\t(omega_1=0 (no relax on isoQ2Q1))",
#                   (0.48, 0.81, 0.00, (1,1,1)),
#                   (0.80, 0.85, 0.00, (2,2,1)),
                  ]:

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

        for cycle_type in ['W', 'V']:
            # assemble/change mnultigrid parameters
            if first_time:
                gmg_lo_obj = iso_mg((stokes_lo, stokes_ho), mg_lvls, inner_cycle, outer_cycle, cycle_type=cycle_type)
                first_time = False
            else:
                gmg_lo_obj.update_params(inner_cycle, outer_cycle, cycle_type=cycle_type)

            gmg_lo     = gmg_lo_obj.get_solver()

            def fsolver(b, x0):
                x, resid_lo  = solve(stokes_ho.A, b, x0=x0, M=gmg_lo,
                                     maxiter=MAXITER, tol=TOL, GMRES=GMRES)
                return resid_lo

            resid_lo    = collect_residuals(fsolver, stokes_ho, plot=False, max_runs=55)
            cf_lsq, itc = conv_factor(resid_lo, multi=True)
            print('\t\t%s\tparams=%s\trho=%.4f\titers=%d' % (cycle_type, str(params), cf_lsq, itc))