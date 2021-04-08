import pyamg
import numpy as np
from scipy.sparse import csr_matrix
from .geometric.taylorhood_interp import construct_interp
from .geometric.full_weight_interp import full_weight_interp_main

def scalar_mg(diffusion, bcs=False, interp_order=1, element_order=1, LEVELs=2, \
              coarsenining_rate=(2,2),
              smoother_type = ('gauss_seidel', {'sweep':'symmetric', 'iterations':1}), \
              DEBUG=False):

    Ain = None
    bcs = None
    if bcs:
        Ain       = diffusion.A
        bcs       = diffusion.bcs
    else:
        Ain       = diffusion.A_no_bcs

    mesh          = diffusion.mesh
    element_order = diffusion.order
    quadrilateral = diffusion.quadrilateral


    vdofs          = diffusion.dof_coord.T
    loc_to_dof_map = np.zeros_like(vdofs, dtype=int)
    dx             = min(vdofs[1][np.nonzero(vdofs[1])])
    dy             = min(vdofs[0][np.nonzero(vdofs[0])])

    # compute index of each element in each direction
    loc_to_dof_map[0] = np.around(vdofs[0]/dy).astype(int)
    loc_to_dof_map[1] = np.around(vdofs[1]/dx).astype(int)

    vDOFs_per_side_x    = diffusion.NE[1]*2+1
    vDOFs_per_side_y    = diffusion.NE[0]*2+1
    # store the DOF at each respective node
    v_fine_grid       = np.zeros((vDOFs_per_side_x, vDOFs_per_side_y), dtype=int)
    for p in range(len(loc_to_dof_map[0])):
        v_fine_grid[loc_to_dof_map[1][p], loc_to_dof_map[0][p]] = p


    Ps, Rs, _, _, _ = construct_interp(diffusion.dof_coord.T, diffusion.bcs_nodes,\
                                       interp_order, element_order,  LVLs=LEVELs,\
                                       quadrilateral=diffusion.quadrilateral, \
                                       coarsening_rate=(2,2), DEBUG=False)

    #######################################################################################
    # Construct Multigrid Hierarchy
    ########################################################################################
    from pyamg.multilevel import multilevel_solver
    levels = []
    levels.append(multilevel_solver.level())


    bcs = diffusion.bcs_nodes
    Ain = Ain.toarray()
    Ain[bcs,:] = Ain[:, bcs] = 0
    import scipy.sparse as sp
    Ain = sp.csr_matrix(Ain)


    levels[-1].A = Ain

    for lvl in range(LEVELs-1):
        ############################################################
        # Set Matrices
        A = levels[-1].A

        ############################################################
        # Interpolation
        P =  csr_matrix(Ps[lvl]) #*0.25
        R =  csr_matrix(Ps[lvl]).T
        # Assemble P
        levels[-1].P  = P #csr_matrix(Ps[lvl])
        # Assemble R
        levels[-1].R  = R #csr_matrix(Rs[lvl])

        ############################################################
        # Coarse & Fine Grids
        levels.append(multilevel_solver.level())
        A = R * A * P                                 # Galerkin operator
        levels[-1].A = A

    ml = multilevel_solver(levels)

    presmoother  = smoother_type
    postsmoother = smoother_type
    pyamg.relaxation.smoothing.change_smoothers(ml, presmoother, postsmoother)


    post = ml.levels[0].postsmoother

    def new_post(A,x,b, *args):
        post(A, x, b)
        x[diffusion.bcs_nodes] = 0

    ml.levels[0].postsmoother = new_post

    pre  = ml.levels[0].presmoother
    def new_pre(A,x, b, *args):
        b[diffusion.bcs_nodes] = 0
        x[diffusion.bcs_nodes] = 0
        pre(A, x, b)
    ml.levels[0].presmoother = new_pre
    def coarse_solve(A,b):
        return np.linalg.pinv(ml.levels[-1].A.toarray())@b

    ml.coarse_solver = coarse_solve

    return ml
