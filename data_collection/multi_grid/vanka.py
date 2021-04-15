from collect import collect

#####################################
# Solver Params
solver_types = [{'module': "stationary",        "resid": 'abs'},
                {'module': ("pyamg", 'fgmres'), "resid": 'abs'}]

cycle_types  = ['V', 'W']
mg_lvls      = 5
NEx          = 64

gmg_params  =  ["Table 2:",
                   "\t(no post-smoothing)",
                   ((1,0,1), 0.86, 0.78, 1.01),
                   ((1,0,2), 0.90, 0.98, 0.83),
                   ((1,0,3), 0.85, 1.02, 0.82),
                   "\t(symmetric)",
                   ((1,1,1), 0.99, 0.67, 0.93),
                   ((1,1,2), 1.05, 0.74, 0.78),
                   ((2,2,1), 0.98, 0.71, 1.05),
                   "Table 3:",
                   "\t(omega_0=0 (no relax on Q2Q1))",
                   ((0,0,1), 0.64, 0.00, 0.98),
                   ((0,0,2), 0.66, 0.00, 0.63),
                   ((1,1,2), 0.46, 0.00, 0.83),
                   "\t(omega_1=0 (no relax on isoQ2Q1))",
                   ((1,1,1), 0.48, 0.81, 0.00),
                   ((2,2,1), 0.80, 0.85, 0.00),
                  ]

collect( ('Vanka', gmg_params),
        NEx=NEx,  problems=['periodic', 'Washer'],
        solver_types = solver_types,
        cycle_types  = cycle_types, mg_lvls=mg_lvls)

