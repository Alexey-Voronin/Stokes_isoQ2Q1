import sys
sys.path.append('../..')
from sysmg import *
from sysmg.util.data_analysis import conv_factor, collect_residuals

#####################################
# system
NEx          = 16
stokes_problems = dict()
print('------------------------------------------------')
print('System set-up')
for BC in ['Washer' ]:
    print('-%s' % BC)
    stokes_problems[BC] = form_stokes(Nelem=(NEx,NEx), domain= (1,1), lo_fe_precond= True,
                                      Solve= False, quadrilateral= True, BCs=BC)

# Solver Params
GMRES        = {'module': "vcycle", "resid": 'abs'}
TOL          = 1e-10
mg_lvls      = 2
MAXITER      = 50

# sample points
N            = 10
OMEGA_OUTER  = np.linspace(0.02, 1., N)
OMEGA_INNER  = np.linspace(0.02, 1., N)

print('\n\nCollect Data:\n')
for BC in ['Washer']:
    stokes_ho    = stokes_problems[BC]
    stokes_lo    = stokes_ho.lo_fe_sys
    # only need to update the entire solver when system changes
    gmg_lo_obj   = None
    first_time   = True
    rho          = np.zeros((N, N)) # (\omega_0, \omega_1)

    for i, omega_outer in enumerate(OMEGA_OUTER):
        for j, omega_inner in enumerate(OMEGA_INNER):
            gamma           = 2
            (nu_1, nu_2)    = (1,1)
            tau             = 1.05
            cycle_type      = 'V'
            smoother_inner  = ('Vanka', {'type'  : 'geometric', 'update': 'additive',
                                         'vcycle':  (1, 1),     'omega_g' : omega_inner})
            inner_cycle     =  dict(multigrid_type    = 'geometric', interpolation_type = (1,1),
                                    coarsenining_rate = (2, 2),               Q2_iso_Q1 = True,
                                    tau      = tau,
                                    gamma    = gamma,
                                    smoother = smoother_inner)
            outer_cycle     = ('Vanka', {'type'  : 'geometric',  'update' : 'additive',
                                         'vcycle':  (nu_1, nu_2), 'omega_g' : omega_outer})
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

            resid_lo    = collect_residuals(fsolver, stokes_ho, plot=False, max_runs=20)
            cf_lsq, itc = conv_factor(resid_lo, multi=True)
            rho[i,j]    = cf_lsq
            print('%.2f\t%.2f\t%.2f' % (omega_outer, omega_inner, cf_lsq))

    np.save("vanka_sens_%s.npy" % BC.lower(), rho)
