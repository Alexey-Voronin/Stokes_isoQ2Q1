import numpy as np
import firedrake
from .taylorhood_interp           import *

def geometric_interp(stokes,  coarsening_rate=(2,2), interpolation_order=(1,1)):
    """Construct geometric interpolation operators for Stokes system."""
    #print('geometric_interp')
    coarsening_rate_x = coarsening_rate[0]
    coarsening_rate_y = coarsening_rate[1]
    ########################################################################################
    # Construct Interpolation operators
    ########################################################################################
    Ps_all         = dict()

    for name, dof_grid_hier, dof_ext_grid_hier, interp_order, bc_hier in zip(
                                           ['Velocity DOFs', 'Pressure DOFs'],
                                           [stokes.v_grid_hier, stokes.p_grid_hier],
                                           [stokes.v_ext_grid_hier, stokes.p_ext_grid_hier],
                                           [interpolation_order[0], interpolation_order[1]],
                                           [stokes.bcs_v_nodes_hier.copy(),    stokes.bcs_p_nodes_hier.copy()]
                                           ):

        Ps = construct_interp(dof_grid_hier, dof_ext_grid_hier,
                              bc_hier, interp_order=interp_order,
                              quadrilateral=stokes.quadrilateral,
                              coarsening_rate=(coarsening_rate_x, coarsening_rate_y),
                              periodic=stokes.periodic)
        Ps_all[name]         = Ps

    return Ps_all
