from collect_scale import collect

#####################################
# Solver Params
solver_type = {'module': ("pyamg", 'fgmres'), "resid": 'abs'}
cycle_types  = ['V', 'W']

gmg_params  = [
                 ((2,2,1), 0.80, 0.85, 0.00),
                 ((1,1,2), 1.05, 0.74, 0.78),
                 ((2,2,1), 0.98, 0.71, 1.05),
               ]

collect(('Vanka', gmg_params),
        problems=['bfs'],
        solver_type= solver_type,
        cycle_types=cycle_types
        )

