import scipy.sparse as sp
import numpy        as np
import matplotlib.pyplot      as plt
import gc


class Vanka(object):
    """
    Vanka relaxation for 2x2 block matrices.

    Vanka relaxation designed for Q2/Q1 and Q1isoQ2/Q1 Stokes Discretizations.

    """

    A = None
    M = None
    B = None
    P = None

    Aloc_inv       = None # dense
    scaled_Aloc_inv_full  = None # block-diagonal csr
    V_neigh_gIDs   = None
    p_gIDs         = None
    v_id_count     = None
    v_scale        = None
    omega          = None
    relax_iters    = None
    vanka_type     = None
    update_type    = None

    #@profile
    def __init__(self, stokes,
                params = {'vcycle': (1,1), 'type' : 'geometric',
                          'update': 'additive', 'omega' : 1,  'debug' : True},
                leg='down',
                Vanka_given=None):
        """
        Vanka initialization function.

        Args:
            stokes (Stokes): stokes object from sysmg.systems.stokes.

            params (dict): dictionary object specifying all the parameters
                needed for BS relaxation.

                Options:
                'vcycle'          : tuple
                    (# of pre-, #  of post-) relaxation sweeps.
                'type'            : {'geometric', 'algebraic'}
                    Vanka relaxation type.
                'update': {'addtive', 'addtive-opt', 'multiplicative'}
                    Update type.
                'omega_g'         : float
                    global update damping.

                Example:
                    {'type'  : 'geometric',  'update' : 'additive',
                     'vcycle': (nu_1, nu_2), 'omega_g': omega_0}

            leg : {'down', 'up'}
                Specifies whether pre- or post- relaxation is being setup.

            Vanka_given: sysmg.relaxation.vanka.Vanka
                Uses already setup Vanka data to save time, since the patches
                are the same for pre- and post- relaxation.

        Returns:
            Nothing

        """
        debug = params['debug'] if 'debug' in params.keys() else False
        if debug:
            print('vanka-init-start-------------------------------------------')

        self.stokes         = stokes
        self.A              = stokes.A
        self.M              = stokes.M
        self.B              = stokes.B if type(stokes.B) == sp.csr_matrix else stokes.B.tocsr()
        self.omega_g        = params["omega_g"]
        self.update_type    = params["update"]
        self.vanka_type     = params["type"]
        self.relax_iters    = params['vcycle'][0] if leg == 'down' else params['vcycle'][1]
        #########################################################
        # for each pressure DoF, find corresponding velocity patch

        # if Vanka has already been setup use it instead of setting ti up anew.
        if Vanka_given is not None:
            if  self.update_type == 'additive-opt':
                self.P = Vanka_given.P
                self.scaled_Aloc_inv_full = Vanka_given.scaled_Aloc_inv_full
            else:
                self.Aloc_inv     = Vanka_given.Aloc_inv
                self.V_neigh_gIDs = Vanka_given.V_neigh_gIDs
                self.p_gIDs       = Vanka_given.p_gIDs
                self.v_id_count   = Vanka_given.v_id_count
                self.v_scale      = Vanka_given.v_scale

            return # nothing else to do

        self.Aloc_inv       = []
        self.V_neigh_gIDs   = []
        self.p_gIDs         = []

        vlen                = self.M.shape[0]
        if self.vanka_type == 'geometric':
            #  DoF relational grids
            v_dofs_grid     = stokes.v_grid_hier[-1]
            tmp             = stokes.v_ext_grid_hier[-1]
            v_dofs_grid_ext = tmp[1:-1,1:-1]
            p_dofs_grid     = stokes.p_grid_hier[-1]
            BC              = 'periodic' if self.stokes.periodic else None

            if debug:
                print('BC=', BC)
                print('v_dofs_grid:\n', v_dofs_grid_ext[-15:, -15:])
                print('p_dofs_grid:\n', p_dofs_grid)

            #######################################################
            Aloc_all = []
            for i in range(v_dofs_grid.shape[0]-1, -1, -2):
                for j in range(v_dofs_grid.shape[1]-1, -1, -2):
                    ip           = i // 2
                    jp           = j // 2
                    p_idx        = p_dofs_grid[ip, jp]
                    if p_idx < 0:
                        continue

                    # figure out velocity patch
                    grid_loc     = v_dofs_grid_ext[i:(i+5),j:(j+5)]

                    v_dofs_loc_x = np.ravel(grid_loc[grid_loc >= 0])
                    v_dofs_loc_y = v_dofs_loc_x+vlen//2
                    v_dofs_loc   = np.array(list(v_dofs_loc_x)+list(v_dofs_loc_y), dtype=int)

                    if debug:
                        print('(i,j)->', i, j)
                        print('p_idx=',p_idx)
                        print('grid_loc:\n', grid_loc)
                        print('v_dofs_loc_x=', v_dofs_loc_x)

                        self.plot_patch(p_idx, v_dofs_loc)
                        print('--------------------------------\n\n')

                    # exctact related matrix entries
                    v_ids        = v_dofs_loc
                    Mloc         = self.M[v_ids,:][:,v_ids]
                    Bloc_csr     = self.B[p_idx,:][:, v_ids]
                    # construct assembled local matrix
                    Aloc         = sp.bmat([[Mloc,      Bloc_csr.T],
                                            [Bloc_csr,      None]]).toarray()
                    Aloc_all.append(Aloc)

                    # save local matrices
                    self.p_gIDs.append(p_idx)
                    self.V_neigh_gIDs.append(v_ids)
                    # all periodic 2x2 element patches are the same for Stokes
                    if BC == 'periodic' and len(self.Aloc_inv) > 0:
                        tmp = np.linalg.norm(np.ravel(Aloc-Aloc_all[0]))
                        if tmp > 1e-10:
                            print('diff=', np.linalg.norm(np.ravel(Aloc-Aloc_all[0])))
                        #else:
                        self.Aloc_inv.append(self.Aloc_inv[-1])
                    else:
                        self.Aloc_inv.append(np.linalg.pinv(Aloc))

            self.V_neigh_gIDs    = self.V_neigh_gIDs
            self.Aloc_inv        = self.Aloc_inv
            self.p_gIDs          = self.p_gIDs

        else: # algebraic
            for p_idx in range(self.B.shape[0]):
                #construct local block matrices
                v_ids    = self.B[p_idx,:].indices
                Mloc     = self.M[v_ids,:][:,v_ids]
                Bloc_csr = self.B[p_idx,:][:, v_ids]

                # construct assembled local matrix
                Aloc     = sp.bmat([[Mloc,      Bloc_csr.T],
                                    [Bloc_csr,      None]]).toarray()

                # save local matrices
                self.p_gIDs.append(p_idx)
                self.V_neigh_gIDs.append(v_ids)
                self.Aloc_inv.append(np.linalg.pinv(Aloc))

                if debug: # plot patches
                    vdof_cord = stokes.v_dof_coord_hier[-1]
                    pdof_cord = stokes.p_dof_coord_hier[-1]

                    nvx     =  self.M.shape[1]//2
                    v_ids_x = v_ids[v_ids < nvx]
                    v_ids_y = v_ids[v_ids >= nvx]
                    v_ids_y = np.array([i % (nvx) for i in v_ids_y])

                    plt.scatter(vdof_cord[:,0], vdof_cord[:,1], marker='o')
                    plt.scatter(vdof_cord[v_ids_x,0], vdof_cord[v_ids_x,1])
                    plt.scatter(vdof_cord[v_ids_y,0], vdof_cord[v_ids_y,1], marker='x')
                    plt.scatter(pdof_cord[p_idx,0], pdof_cord[p_idx,1])
                    plt.show()
        ##########################################################
        # see in how many patches velocity DoFs appear
        self.v_id_count = np.zeros((self.M.shape[0],), dtype=int)
        for v_id_patch in self.V_neigh_gIDs:
            self.v_id_count[v_id_patch] +=1

        if debug: # check to see that v_id_count makes sense
            vdof_cord = stokes.v_dof_coord_hier[-1]
           # vdof_cord = vdof_cord[vdof_cord >= 0]
            nx        = self.M.shape[1]//2
            for i in range(len(self.v_id_count)):
                plt.annotate(str(self.v_id_count[i]), xy=(vdof_cord[i%nx,0], vdof_cord[i%nx,1]), fontsize =10)
            plt.show()

        self.v_scale = []
        for pID, v_id_patch in zip(self.p_gIDs, self.V_neigh_gIDs):
            self.v_scale.append( np.array(1. / self.v_id_count[v_id_patch]))
            assert not np.any(np.isnan(self.v_scale[-1])), 'There is NaN/inf in v_scale'

        if self.update_type == 'additive-opt':
            # compute total length of vectors with DoFs split up
            ntotal = len(self.p_gIDs)
            for v_ids in self.V_neigh_gIDs:
                ntotal += len(v_ids)

            # construct gather/scatter operator to map betweem
            # assembled and and dis-assembled matrix
            rows = np.linspace(0, ntotal-1, ntotal, dtype=int)
            cols = np.zeros((ntotal,))
            col_count = 0
            for v_ids, p_id in zip(self.V_neigh_gIDs, self.p_gIDs):
                for j, v in enumerate(v_ids):
                    cols[col_count+j] = v
                col_count += len(v_ids)
                cols[col_count] = self.M.shape[0]+p_id
                col_count += 1

            self.P = sp.coo_matrix((np.ones((ntotal,)), (rows, cols)), shape=(ntotal, self.A.shape[1])).tocsr()
            del self.V_neigh_gIDs
            del cols
            del rows
            gc.collect()

            # compute block-diagonal inversion and scaling matrix
            scaled_Aloc_inv = []
            scale_max       = np.zeros((51,))
            total_values = 0
            for scale, Aloc_inv in zip(self.v_scale, self.Aloc_inv):
                scale_max[:len(scale)] = scale
                scale_max[len(scale)]  = 1
                scaled_Aloc_inv.append(np.diag(scale_max[:(len(scale)+1)])@Aloc_inv)

                total_values += scaled_Aloc_inv[-1].shape[0]*scaled_Aloc_inv[-1].shape[1]
            # quiet slow..
            self.scaled_Aloc_inv_full = sp.block_diag(scaled_Aloc_inv, format="csr")

            self.scaled_Aloc_inv_full.sort_indices()
            self.P.sort_indices()

            if self.stokes.periodic:
                self.scaled_Aloc_inv_full = sp.bsr_matrix(self.scaled_Aloc_inv_full, blocksize=Aloc_inv.shape)

            del self.Aloc_inv
            del scaled_Aloc_inv
            del self.v_scale
            gc.collect()

        if debug:
            print('vanka-init-end-------------------------------------------')


    #@profile
    def relax(self, A, x, b):
        """
        Apply Vanka relaxation.

        Args:
            A (csr_matrix):
                Systems matrix (ignored, but needed to be compatible with pyamg relaxation).

            x (numpy.ndarray):
                Initial guess.

            b: (numpy.ndarray)
                Right-hand side.

            alpha: (float)
                optional parameter to change alpha variable/

            callback: (function)
                records the residual at each iteration of relax.

        Returns:
            (numpy.ndarray):  Solution vector.

        Notes:
            If you would like to chain parameters then modify the relaxation object via the helper functions.

        """
        vlen = self.M.shape[0]
        x    = x.copy()
        u    = x[0:vlen]
        p    = x[vlen:]

        if self.update_type == 'multiplicative':
            # relaxation iteratioons
            for i in range(self.relax_iters):
                for p_id, v_ids, Aloc_inv, v_scale in zip(self.p_gIDs, self.V_neigh_gIDs,
                                                          self.Aloc_inv, self.v_scale):
                    # recompute residual and split it
                    r     = b-A*x
                    ru    = r[:vlen]
                    rp    = r[vlen:]

                    #v_ids = self.V_neigh_gIDs[p_id]
                    rloc  = np.hstack((ru[v_ids],
                                       rp[p_id]))
                    # compute patch solution
                    dx_loc= Aloc_inv@rloc
                    du_loc= dx_loc[:-1]
                    dp_loc= dx_loc[-1]

                    assert not np.any(np.isnan(dx_loc)), print('dx_loc has NaN:',dx_loc)

                    # update current global solution
                    u[v_ids]    += self.omega_g*du_loc*v_scale
                    p[p_id]     += self.omega_g*dp_loc

        else: #additive
            if self.update_type == 'additive-opt':
                for i in range(self.relax_iters):
                    r       = b-A*x
                    # split residual into blocks
                    r_split = self.P*r
                    # compute correction block-wise and then assemble it
                    x      += self.omega_g*(self.P.T*(self.scaled_Aloc_inv_full*r_split))
                    """
                    # for profiling
                    unassembled_sol = self.scaled_Aloc_inv_full*r_split
                    assembled_sol   = self.P.T*unassembled_sol
                    x              += self.omega_g*assembled_sol
                    """
            else: # old implementation
                rloc = np.zeros((51,)) # assumes max size of Q2-Q1 patch (25*2+1)
                for i in range(self.relax_iters):
                     # recompute residual and split it
                     r     = b-A*x
                     curr_resid = np.linalg.norm(b - A * x)
                     ru    = r[:vlen]
                     rp    = r[vlen:]
                     for p_id, v_ids, Aloc_inv, v_scale in zip(self.p_gIDs,   self.V_neigh_gIDs,
                                                              self.Aloc_inv, self.v_scale):

                         ru_vids = ru[v_ids]
                         rp_pid  = rp[p_id]

                         true_size           = len(ru_vids)+1
                         rloc[:len(ru_vids)] = ru_vids
                         rloc[true_size-1]   = rp_pid

                         # compute patch solution
                         dx_loc= Aloc_inv@rloc[:true_size]
                         du_loc= dx_loc[:-1]
                         dp_loc= dx_loc[-1]

                         # update current global solution
                         u[v_ids]   += self.omega_g*du_loc*v_scale
                         p[p_id]    += self.omega_g*dp_loc

        return x

    def plot_patch(self, p_idx, v_ids):
        vdof_cord = self.stokes.v_dof_coord_hier[-1]
        pdof_cord = self.stokes.p_dof_coord_hier[-1]

        nvx       = self.M.shape[1] // 2
        v_ids_x   = v_ids[v_ids < nvx]
        v_ids_y   = v_ids[v_ids >= nvx]

        plt.scatter(vdof_cord[:, 0], vdof_cord[:, 1], marker='o')
        plt.scatter(vdof_cord[v_ids_x, 0], vdof_cord[v_ids_x, 1])
        plt.scatter(pdof_cord[:, 0], pdof_cord[:, 1], marker='x')
        plt.scatter(pdof_cord[p_idx, 0], pdof_cord[p_idx, 1])
        plt.show()

    def print_params(self):
        print('omega=%.2f, iters=%d' % (self.omega_g, self.relax_iters))

    def update_omega_g(self,omega_g):
        self.omega_g = omega_g

    def update_relax_iters(self, relax_iters):
        self.relax_iters = relax_iters

    def relax_inplace(self, A, x, b):
        """
        Apply inplace Vanka relaxation.

        Args:
            A: csr_matrix
                Systems matrix (ignored, but needed to be compatible with pyamg
                relaxation).

            x: numpy.ndarray
                Initial guess.

            b: numpy.ndarray
                Right hand side.

        Returns:
            None, x will be modified in place.

        Notes:
            If you would like to chain parameters then modify the relaxation
            object via the helper functions.

        """
        x[:] = self.relax(A, x, b)
