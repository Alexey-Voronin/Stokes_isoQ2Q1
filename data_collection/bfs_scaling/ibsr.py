from collect_scale import collect

#####################################
# Solver Params
solver_type = {'module': ("pyamg", 'fgmres'), "resid": 'abs'}
cycle_types  = ['V', 'W']

gmg_params  = [
#                "Table 2:",
#               "\t(no post-smoothing)",
#               ((1,0,1), 0.87, 1.02, 0.90, 1.20, 0.79, 0.74, 0.88),
#               ((1,0,2), 0.86, 0.75, 1.04, 0.99, 0.70, 1.15, 0.88),
#               "\t(symmetric)",
               ((1,1,1), 0.97, 0.91, 1.04, 1.02, 0.93, 0.78, 0.86),
               ((2,2,1), 1.04, 0.55, 0.59, 0.72, 0.99, 0.85, 1.30),
#               "Table 3:",
#               "\t(omega_0=0 (no relax on Q2Q1))",
#               ((0,0,1), 0.93, 0.00, 0.24, -1.0, -1.0, 0.67, 1.56),
#               ((0,0,2), 0.63, 0.00, 0.40, -1.0, -1.0, 0.52, 1.35),
#               "\t(omega_1=0 (no relax on isoQ2Q1))",
#               ((1,1,1), 0.77, 0.45, 0.00, 0.46, 1.37, -1.0, -1.0),
#               ((2,2,1), 0.99, 0.85, 0.00, 0.90, 1.12, -1.0, -1.0)
               ]

collect(('IBSR', gmg_params),
        problems=['bfs'],
        solver_type= solver_type,
        cycle_types=cycle_types
        )

