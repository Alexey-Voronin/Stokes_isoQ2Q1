import scipy.sparse as sp
from .taylorhood_interp_stencils import getInterpStencils

def construct_interp(dof_grid_hier, dof_ext_grid_hier, bc_hier, interp_order,
                     quadrilateral=True, coarsening_rate=(2,2),  periodic=False):
    """Construct interpolation opertor for fields discritized with Q1 or Q2 elements."""
    ###################################################################
    Ps            = []
    if coarsening_rate[0] == 2 and coarsening_rate[1] == 2:
        stencils = getInterpStencils(interp_order, quadrilateral=quadrilateral)
    else:
         raise ValueError('coarsening_rate=%s is not supported atm' % str(coarsening_rate))

    # Construct interpolation between all the grids
    for lvl in range(len(dof_grid_hier)-1, 0, -1):
        #########################################################################
        #  DOF relational map
        bcs                 = bc_hier[lvl - 1]
        cdofs_grid          = dof_grid_hier[lvl-1]
        fdofs_grid          = dof_grid_hier[lvl]
        extended_dofs_grid  = dof_ext_grid_hier[lvl]

        PT_rows = []; PT_cols = []; PT_data = []
        # obtain the interpolation stencils
        interp_stencil_p1 = None
        interp_stencil_p2_node = interp_stencil_p2_hside =  \
                interp_stencil_p2_vside = interp_stencil_p2_center = None
        if interp_order == 1:
            interp_stencil_p1 = stencils['nodal'][:,::-1] # correction for renumbering
        elif interp_order == 2:
            interp_stencil_p2_node   = stencils['nodal'][:,::-1]
            interp_stencil_p2_hside  = stencils['horizontal'][:,::-1]
            interp_stencil_p2_vside  = stencils['vertical'][:,::-1]
            interp_stencil_p2_center = stencils['center'][:,::-1]

        if interp_order == 1:
            stencil_h_offset = 2
        else:
            stencil_h_offset = 0

        # restrict solution to coarse grid
        cdofs_per_xdim, cdofs_per_ydim = cdofs_grid.shape
        # DO NOT DELTE -
        # The reason for coarsening "from the end" is due to periodic mesh layout
        #           fdofs_grid:                    cdofs_grid:
        # [[ 0  8 16 24 32 40 48 56]            [[ 0  4  8 12]
        #  [ 1  9 17 25 33 41 49 57]            [ 1  5  9 13]
        # [ 2 10 18 26 34 42 50 58]   <-->      [ 2  6 10 14]
        # [ 3 11 19 27 35 43 51 59]             [ 3  7 11 15]]
        # [ 4 12 20 28 36 44 52 60]
        # [ 5 13 21 29 37 45 53 61]
        # [ 6 14 22 30 38 46 54 62]
        # [ 7 15 23 31 39 47 55 63]]
        #
        # Note: dof 3 in cdofs_grid is actually dof 14 in fdofs_grid.
        # When coarsening from the end I am able to avoid keeping track of this


        for jc in range(cdofs_per_ydim-1, -1, -1):
            for ic in range(cdofs_per_xdim-1, -1,-1):

                coarse_dof = cdofs_grid[ic, jc]
                if cdofs_grid[ic, jc] == -1:
                    continue

                ifine = ic * coarsening_rate[0] + stencil_h_offset
                jfine = jc * coarsening_rate[1] + stencil_h_offset
                if periodic: # check note above
                    ifine       += 1
                    jfine       += 1

                if interp_order == 1:
                    target_height = interp_stencil_p1.shape[0]
                    target_width  = interp_stencil_p1.shape[1]
                    dof_patch     = extended_dofs_grid[ifine:(ifine+target_height),
                                                       jfine:(jfine+target_width)]

                    for idx in range(dof_patch.shape[0]):
                        for idy in range(dof_patch.shape[1]):
                            fine_dof = dof_patch[idx, idy]
                            if fine_dof == -1: # padding
                                continue

                            if coarse_dof not in bcs:
                                PT_rows.append(coarse_dof)
                                PT_cols.append(fine_dof)
                                PT_data.append(interp_stencil_p1[idx, idy])

                elif interp_order == 2:
                    padi            = 0
                    padj            = 0
                    weights_stencil = None

                    if (ic % 2) + (jc % 2) == 0:
                        stencil_type    = 'nodal'
                        weights_stencil = interp_stencil_p2_node
                    elif (ic % 2) + ((jc+1) % 2) == 0:   # horizontal edge
                        weights_stencil = interp_stencil_p2_hside
                        stencil_type    ='horizontal'
                        padi            = stencil_h_offset
                        padj            = 2
                    elif ((ic+1) % 2) + (jc % 2) == 0:     # vertical edge
                        stencil_type    ='vertical'
                        weights_stencil = interp_stencil_p2_vside
                        padj            = stencil_h_offset
                        padi            = 2
                    elif ((ic+1) % 2) + ((jc+1) % 2) == 0: # hypotenuse
                        stencil_type    = 'center'
                        weights_stencil = interp_stencil_p2_center
                        padj            = 2
                        padi            = 2
                    else:
                        print('Stencil Error: you messed up')

                    # need to adjust for padding
                    ifine       += padi
                    jfine       += padj

                    dof_patch = extended_dofs_grid[ifine:(ifine+weights_stencil.shape[0]),
                                                   jfine:(jfine+weights_stencil.shape[1])]

                    for idx in range(dof_patch.shape[0]):
                        for idy in range(dof_patch.shape[1]):
                            fine_dof = dof_patch[idx, idy]
                            if fine_dof == -1: # padding
                                continue

                            if  coarse_dof not in bcs:
                                PT_rows.append(coarse_dof)
                                PT_cols.append(fine_dof)
                                PT_data.append(weights_stencil[idx, idy])
                else:
                    print('ORDER Error: do not have thje right stencil for these elements' )

        coarse_dofs = len(cdofs_grid[cdofs_grid >=0])
        fine_dofs   = len(fdofs_grid[fdofs_grid >=0])
        PT          = sp.coo_matrix((PT_data, (PT_rows, PT_cols)), shape=(coarse_dofs, fine_dofs)).tocsr()
        Ps.append(PT.T)

    return Ps[::-1]   # renumber to match other data hierarchies
