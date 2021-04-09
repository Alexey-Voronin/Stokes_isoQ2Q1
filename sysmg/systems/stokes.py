from firedrake import *
import numpy as np
import scipy.sparse as sp
from scipy.sparse import bmat
import matplotlib.pyplot as plt
from .stokes_util.stokes_periodic import *


class StokesSystem(object):
    """
    Stokes systems object.

    Constructs hierarchy of 2 by 2 Stokes system, where each block operator is a sparse matrix.
    In addition stores complementary geometrical data strcutures (firedrake meshes,
    locations of DoFs, etc).

    """

    domain                  = None
    mesh                    = None
    mesh_hierarchy          = None
    p_fxn_space             = None
    v_fxn_space             = None
    quadrilateral           = None
    isoQ2_Q1                = None

    bcs_v_hier              = None
    bcs_p_hier              = None
    bcs_v_nodes_hier        = None
    bcs_p_nodes_hier        = None

    dof_ordering            = None
    periodic                = None
    lo_fe_precond           = None
    NEx_hier                = None
    NEy_hier                = None
    A                       = None
    A_without_bc            = None
    B_without_bc            = None
    BT_without_bc           = None
    M_without_bc            = None
    A_bcsr                  = None
    b                       = None
    M                       = None
    BT                      = None
    B                       = None
    Zero_block              = None  # not actually zero when BCs imposed on Pressure
    Mass_v                  = None
    Mass_p                  = None
    Pressure_stiff          = None
    nullspace               = None
    upsol                   = None
    usol                    = None
    psol                    = None
    P                       = None  # nodal order --> component wise dof_ordering (for entire system)
    Pv_split                = None
    P_split                 = None
    P_renumber              = None
    Pv_sort                 = None
    Pp_sort                 = None

    v_order                 = None
    p_order                 = None
    v_dof_coord_hier        = None
    p_dof_coord_hier        = None
    dof_map_v_hier          = None
    dof_map_p_hier          = None
    v_grid_hier             = None
    p_grid_hier             = None
    v_ext_grid_hier         = None
    p_ext_grid_hier         = None

    v_space_hier            = None
    p_space_hier            = None
    vp_space_hier           = None

    upsol_firedrake         = None
    lo_fe_sys               = None
    upsol_firedrake         = None

    def __init__(self):
        """
        Initialize data structures.

        Creates a Stokes objects with empty data fields. This object is then
        populated either by calling sysmg.systems.stokes.form_stokes function,
        or is used to store coarse systems constructed by multigrid solvers
        in sysmg.multigrid.

        Args:
            None.

        Returns:
            Nothing.

        """
        self.NEx_hier           = []
        self.NEy_hier           = []
        self.v_space_hier       = []
        self.p_space_hier       = []
        self.vp_space_hier      = []
        self.v_dof_coord_hier   = []
        self.p_dof_coord_hier   = []
        self.bcs_v_hier         = []
        self.bcs_p_hier         = []
        self.bcs_v_nodes_hier   = []
        self.bcs_p_nodes_hier   = []
        # for GMG an Vanka
        self.v_grid_hier        = []
        self.p_grid_hier        = []
        self.v_ext_grid_hier    = []
        self.p_ext_grid_hier    = []
        self.isoQ2_Q1 = False
        self.periodic = False

    #@profile
    def renumber_dofs(self):
        """
        Renumbers the DoFs of freedom in the Stokes system using lexsort.

        The renumbering is not technically necessary, but it's easier to
        construct GMG interpolation on sorted grids. DoFs maps, Matrices and all
        other complementary data structures are renumbered as well.

        """
        A               = self.A
        M               = self.M
        B               = self.B
        BT              = self.BT
        Mass_v          = self.Mass_v
        Mass_p          = self.Mass_p
        Pressure_stiff  = self.Pressure_stiff
        B_without_bc    = self.B_without_bc
        BT_without_bc   = self.BT_without_bc
        M_without_bc    = self.M_without_bc

        P_sort_all_hier = []
        dof_map_all_hier= []
        for v_dof_coord_hier, p_dof_coord_hier in zip(self.v_dof_coord_hier, self.p_dof_coord_hier):
            P_sort_all = dict()
            dof_map_all = dict()

            for v_coord, var in zip([v_dof_coord_hier, p_dof_coord_hier], ['v_x', 'p']):
                dof = v_coord.T
                if self.periodic:
                    dof     = dof.round(12)
                ind     = np.lexsort((dof[0], dof[1]))

                rows    = []
                cols    = []
                dof_map = dict()
                for j in range(len(ind)):
                    rows.append(j)
                    cols.append(ind[j])
                    dof_map[ind[j]] = j

                Nx = v_coord.shape[0]
                P_sort = sp.coo_matrix((np.ones(len(rows)), (np.array(rows),
                                        np.array(cols))), (Nx, Nx)).tocsr()

                P_sort_all[var] = P_sort
                dof_map_all[var] = dof_map
            P_sort_all_hier.append(P_sort_all)
            dof_map_all_hier.append(dof_map_all)

        # sort coordinates
        for v_dof, p_dof, P_sort_all in zip(self.v_dof_coord_hier, self.p_dof_coord_hier, P_sort_all_hier):
            Pvx         = P_sort_all['v_x']
            Pp          = P_sort_all['p']
            v_dof[:, :] = Pvx * v_dof[:,:]
            p_dof[:, :] = Pp  * p_dof[:,:]

        # sort boundary nodes
        bcs_v_nodes_hier_renum = []
        bcs_p_nodes_hier_renum = []
        for bc_v_nodes, bc_p_nodes, dof_maps in zip(self.bcs_v_nodes_hier, self.bcs_p_nodes_hier, dof_map_all_hier):
            map_v = dof_maps['v_x']
            map_p = dof_maps['p']

            bcs_v_nodes_hier_renum.append([map_v[v] for v in bc_v_nodes])
            bcs_p_nodes_hier_renum.append([map_p[p] for p in bc_p_nodes])

        self.bcs_v_nodes_hier   = bcs_v_nodes_hier_renum
        self.bcs_p_nodes_hier   = bcs_p_nodes_hier_renum
        self.dof_map_v_hier     = [dof_map['v_x'] for dof_map in dof_map_all_hier]
        self.dof_map_p_hier     = [dof_map['p']   for dof_map in dof_map_all_hier]

        P_sort_all      = P_sort_all_hier[-1]
        Pv_sort         = bmat([[P_sort_all['v_x'], None],
                             [None, P_sort_all['v_x']]]).tocsr()
        Pp_sort         = P_sort_all['p']
        P_sort_full     = bmat([[P_sort_all['v_x'], None, None],
                                 [None, P_sort_all['v_x'], None],
                                 [None, None, P_sort_all['p']]]).tocsr()

        self.Pv_sort    = P_sort_all['v_x']
        self.Pp_sort    = Pp_sort

        M_sorted            = Pv_sort *      M * Pv_sort.T
        B_sorted            = Pp_sort *      B * Pv_sort.T
        BT_sorted           = Pv_sort *     BT * Pp_sort.T
        Mass_v_sorted       = Pv_sort * Mass_v * Pv_sort.T
        Mass_p_sorted       = Pp_sort * Mass_p * Pp_sort.T
        Pressure_stiff_sorted=Pp_sort * Pressure_stiff * Pp_sort.T
        # no bcs
        M_without_bc_sorted = Pv_sort * M_without_bc * Pv_sort.T
        B_without_bc_sorted = Pp_sort * B_without_bc * Pv_sort.T
        BT_without_bc_sorted= Pv_sort * BT_without_bc * Pp_sort.T

        # replace with new matrices
        self.A              = P_sort_full * A * P_sort_full.T
        self.M              = M_sorted
        self.B              = B_sorted
        self.BT             = BT_sorted

        self.M_without_bc   = M_without_bc_sorted
        self.B_without_bc   = B_without_bc_sorted
        self.BT_without_bc  = BT_without_bc_sorted

        self.Mass_p         = Mass_p_sorted
        self.Pressure_stiff = Pressure_stiff_sorted
        self.Mass_v         = Mass_v_sorted
        self.b              = P_sort_full * self.b
        self.nullspace      = P_sort_full * self.nullspace
        self.P_renumber     = P_sort_full
        if self.upsol is not None:
            self.upsol = P_sort_full * self.upsol

    def create_split_v_copy(self):
        """
        Reorders DoFs componentwise.

        Splits the vector laplacian and all other relevant data structures
        componentwise. This is necessary to be able to construct multigrid
        solvers.
            Original DoF ordering [vx_1 vy_1 .. vx_n P].
            New split DoF ordering[Vx        Vy      P ].

        """
        stokes_split                    = StokesSystem()
        stokes_split.domain             = self.domain
        stokes_split.mesh               = self.mesh
        stokes_split.mesh_hierarchy     = self.mesh_hierarchy
        stokes_split.quadrilateral      = self.quadrilateral
        stokes_split.dof_ordering       = self.dof_ordering
        stokes_split.bcs_v_hier         = self.bcs_v_hier
        stokes_split.bcs_p_hier         = self.bcs_p_hier
        stokes_split.bcs_v_nodes_hier   = self.bcs_v_nodes_hier
        stokes_split.bcs_p_nodes_hier   = self.bcs_p_nodes_hier
        stokes_split.v_space_hier       = self.v_space_hier
        stokes_split.p_space_hier       = self.p_space_hier
        stokes_split.vp_space_hier      = self.vp_space_hier
        stokes_split.upsol_firedrake          = self.upsol_firedrake

        stokes_split.NEx_hier           = self.NEx_hier
        stokes_split.NEy_hier           = self.NEy_hier
        stokes_split.p_dof_coord_hier   = self.p_dof_coord_hier
        stokes_split.v_dof_coord_hier   = self.v_dof_coord_hier
        stokes_split.dof_map_v_hier     = self.dof_map_v_hier
        stokes_split.dof_map_p_hier     = self.dof_map_p_hier
        stokes_split.v_order            = self.v_order
        stokes_split.p_order            = self.p_order
        stokes_split.periodic           = self.periodic
        stokes_split.isoQ2_Q1           = self.isoQ2_Q1

        nv  = self.M.shape[0]
        nvx = int(nv / 2)

        rows = []; cols = []
        for j in range(0, nvx):
            rows.append(j)
            cols.append(j * 2)
            rows.append(j + nvx)
            cols.append(j * 2 + 1)

        Pv = sp.coo_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))),
                           shape=self.M.shape).tocsr()
        Ip = sp.identity(self.B.shape[0], dtype=float, format='dia')

        stokes_split.P = sp.bmat([[Pv, None],
                                  [None, Ip]])
        P = stokes_split.P

        stokes_split.Mass_v         = Pv * self.Mass_v * Pv.T
        stokes_split.Mass_p         = self.Mass_p
        stokes_split.Pressure_stiff = self.Pressure_stiff
        stokes_split.M              = Pv * self.M * Pv.T
        stokes_split.BT             = Pv * self.BT * Ip
        stokes_split.B              = Ip * self.B * Pv.T
        stokes_split.Zero_block     = self.Zero_block

        stokes_split.M_without_bc   = Pv * self.M_without_bc * Pv.T
        stokes_split.BT_without_bc  = Pv * self.BT_without_bc * Ip
        stokes_split.B_without_bc   = Ip * self.B_without_bc * Pv.T

        if self.nullspace is not None:
            stokes_split.nullspace = P * self.nullspace

        stokes_split.A_bcsr         = sp.bmat([[stokes_split.M, stokes_split.BT],
                                                [stokes_split.B, -1 * stokes_split.Zero_block]])
        stokes_split.A              = stokes_split.A_bcsr.tocsr()

        stokes_split.A_without_bc   = sp.bmat([[stokes_split.M_without_bc, stokes_split.BT_without_bc],
                                             [stokes_split.B_without_bc, None]]).tocsr()

        if self.upsol is not None:
            stokes_split.upsol  = P * self.upsol
            stokes_split.usol   = stokes_split.upsol[:self.M.shape[0]]
            stokes_split.psol   = stokes_split.upsol[self.M.shape[0]:]

        stokes_split.b  = P * self.b
        stokes_split.bu = stokes_split.b[:self.M.shape[0]]
        stokes_split.bp = stokes_split.b[self.M.shape[0]:]

        stokes_split.Pv_split   = Pv
        stokes_split.P_split    = P

        # np.linalg.norm(self.A*self.upsol-stokes_split.P.T*(stokes_split.A*stokes_split.upsol))

        # error = np.linalg.norm(self.A*self.upsol-stokes_split.P.T*(stokes_split.A*stokes_split.upsol))
        # assert (error < 1e-10), "create_split_v_copy: Reordered Matrix might not be correct"

        return stokes_split


    def grid_hierarchy(self):
        """
        Construct 2D arrays summarizing topological location of DoFs.

        Construct 2D arrays containing DoF number for each field (velocity,
        pressure). These grids only make sense for structured grids.
        These grid are later used to construct interpolation operators and
        Vanka relaxation patches.

        """
        bc_type = 'periodic' if self.periodic else 'dirichlet'
        for NEx, NEy, v_dof_coord, p_dof_coord in zip(self.NEx_hier,         self.NEy_hier,
                                                      self.v_dof_coord_hier, self.p_dof_coord_hier):

            ########################################################################################
            # velocity DOFs map
            ########################################################################################
            # print('-velocity DOFs map')
            if self.v_order == 2 or self.lo_fe_precond:
                vDOFs_per_side_x    = (NEx*2+1)
                vDOFs_per_side_y    = (NEy*2+1)
            else:
                vDOFs_per_side_x    = (NEx+1)
                vDOFs_per_side_y    = (NEy+1)

            if self.periodic:
                vDOFs_per_side_x -= 1
                vDOFs_per_side_y -= 1

            vdofs               = v_dof_coord.T
            loc_to_dof_map      = np.zeros_like(vdofs, dtype=int)

            min_val             = np.min(vdofs, axis=1)
            vdof_shift          = (vdofs.T-min_val).round(12).T
            dx                  = min((vdof_shift[0][np.nonzero(vdof_shift[0])]))
            dy                  = min((vdof_shift[1])[np.nonzero(vdof_shift[1])])
            # compute index of each element in each direction
            loc_to_dof_map[1]   = np.round( (vdof_shift[1])/dy).astype(int)
            loc_to_dof_map[0]   = np.round( (vdof_shift[0])/dx).astype(int)


            v_fine_grid         = np.zeros((vDOFs_per_side_x, vDOFs_per_side_y), dtype=int) - 1
            # store the DOF at each respective node
            for p in range(len(loc_to_dof_map[0])):
                v_fine_grid[loc_to_dof_map[0][p], loc_to_dof_map[1][p]] = p

            offset = 3 #if self.v_order == 2 or self.isoQ2_Q1 else 1
            self.v_grid_hier.append(v_fine_grid)
            self.v_ext_grid_hier.append(self.extend_dof_grid(v_fine_grid, bc_type=bc_type, padding=offset))
            ########################################################################################
            # pressure DOFs map
            ########################################################################################
            #print('-pressure DOFs map')
            pDOFs_per_side_x    = (NEx+1)
            pDOFs_per_side_y    = (NEy+1)
            pdofs               = p_dof_coord.T

            if self.periodic:
                pDOFs_per_side_x -= 1
                pDOFs_per_side_y -= 1

            loc_to_dof_map      = np.zeros_like(pdofs, dtype=int)
            min_val             = np.min(pdofs, axis=1)
            pdof_shift          = (pdofs.T-min_val).round(12).T
            dx                  = min((pdof_shift[0][np.nonzero(pdof_shift[0])]))
            dy                  = min((pdof_shift[1])[np.nonzero(pdof_shift[1])])
            # compute index of each element in each direction
            loc_to_dof_map[1]   = np.around( (pdofs[1] - min_val[1])/dy).astype(int)
            loc_to_dof_map[0]   = np.around( (pdofs[0] - min_val[0])/dx).astype(int)

            # store the DOF at each respective node
            p_fine_grid = np.zeros((pDOFs_per_side_x, pDOFs_per_side_y), dtype=int) - 1
            for p in range(len(loc_to_dof_map[0])):
                p_fine_grid[loc_to_dof_map[0][p], loc_to_dof_map[1][p]] = p

           # offset =  1
            self.p_grid_hier.append(p_fine_grid)
            self.p_ext_grid_hier.append(self.extend_dof_grid(p_fine_grid, bc_type=bc_type, padding=offset))

    def extend_dof_grid(self, dofs_grid, bc_type=None, padding=0):

        extended_dofs_grid = None
        if padding > 0:
            pad = padding
            if bc_type == 'periodic':
                extended_dofs_grid = np.ones((dofs_grid.shape[0] + pad * 2,
                                              dofs_grid.shape[1] + pad * 2), dtype=int) * -1
                extended_dofs_grid[pad:-1 * pad, pad:-1 * pad] = dofs_grid

                lx = dofs_grid.shape[1]
                ly = dofs_grid.shape[0]
                # left and right border
                extended_dofs_grid[pad:-1 * pad, :pad] = dofs_grid[:, (lx - pad):(lx - 0)]
                extended_dofs_grid[pad:-1 * pad, (lx + pad):] = dofs_grid[:, :pad]
                # top and bottom
                extended_dofs_grid[:pad, pad:-1 * pad] = dofs_grid[(ly - pad):(ly - 0), :]
                extended_dofs_grid[(ly + pad):, pad:-1 * pad] = dofs_grid[:pad, :]
                # corners
                extended_dofs_grid[:pad, :pad] = dofs_grid[(ly - pad):(ly - 0), (lx - pad):(lx - 0)]
                extended_dofs_grid[(ly + pad):, (lx + pad):] = dofs_grid[:pad, :pad]
                extended_dofs_grid[:pad, (lx + pad):(lx + pad * 2)] = dofs_grid[(ly - pad):(ly - 0), :pad]
                extended_dofs_grid[(lx + pad):(lx + pad * 2), :pad] = dofs_grid[:pad, (ly - pad):(ly - 0)]
            else:
                extended_dofs_grid = np.ones((dofs_grid.shape[0] + pad * 2,
                                              dofs_grid.shape[1] + pad * 2), dtype=int) * -1
                extended_dofs_grid[pad:-1 * pad, pad:-1 * pad] = dofs_grid

        return extended_dofs_grid


#@profile
def form_stokes(Nelem=(8, 8), domain=(1., 1.), quadrilateral=True,
                Q1Q1=False, isoQ2_Q1=False, BCs='Washer',
                lo_fe_precond=False, mesh_verts=None,
                dof_ordering={'split_by_component': True, 'renumber_based_on_coord': True},
                Plot=False, Solve=True,  printstuff=False):
    """
    Construct Stokes system using Taylor-Hood discretizations.

    Args:
        Nelem (optional):  tuple of ints
            Number of elements in each dimension.
        domain (optional): tuple of floats
            Maximum length of topology in each dimension.
        quadrilateral (optional): boolean
            Creates quadrilateral mesh, defaults to True.
        Q1Q1 (optional): boolean
            Uses Q1/Q1 discretization, default to False (Q2/Q1 discretization).
        BCs (optional): str
            Describes domain topology and boundary conditions.

            Options:
                'Washer'      : lid-driven cavity, enclosed flow.
                'periodic'    : square domain with periodic boundary bcs.
                'bfs'         : backward-facing step (L-shaped) domain with
                                parabolic inflow bc, natural outflow bcs.
                'in-out-flow' : square domain with parabolic inflow bc,
                                natural outflow bcs.
        lo_fe_precond (optional): boolean
            Constructs complementary Q1isoQ2/Q1 system, defaults to False.
        mesh_verts (optional): numpy.ndarray
            presribes location of mesh nodes, allowing to construct distorted
            meshes, default to None.
        dof_ordering(optional): dict
            Specifies how DoF in the Matrices and complementary data structures
            will be ordered:
            Example:
                {'split_by_component': True, 'renumber_based_on_coord': True}
        Plot(optional): boolean
            Plot mesh and solution, defaults to false.
        Solve(optional): boolean
            Solve the system, default to false

    Returns:
        sysmg.systems.stokes.StokesSystem : Stokes object


    """
    if printstuff:
        print('form_stokes------------------------------------------------------')
    stokes              = StokesSystem()
    stokes.quadrilateral= quadrilateral
    stokes.domain       = domain
    stokes.lo_fe_precond= lo_fe_precond
    Solve               = False if Q1Q1 or isoQ2_Q1 else Solve
    stokes.isoQ2_Q1     = isoQ2_Q1

    if printstuff:
        print('start mesh setup')

    if BCs == 'periodic':
        if Nelem[0] != Nelem[1]:
            raise ValueError("Number of elements in each directions must be the same")
        stokes.periodic         = True
        stokes.mesh_hierarchy   = []
        for nex in [2 ** i for i in range(2, 1 + int(np.log2(Nelem[0])))]:
            stokes.NEx_hier.append(nex)
            stokes.NEy_hier.append(nex)
            stokes.mesh_hierarchy.append(PeriodicProblem(nex, 0, use_quads=quadrilateral).mesh())  # first arg base, second # of refinements
        mesh = stokes.mesh_hierarchy[-1]

    elif BCs == 'bfs':
        if Nelem[0] != Nelem[1]:
            raise ValueError("Number of elements in each directions must be the same")

        from sysmg.util.firedrake_mesh import RectangleMesh_MyCoord
        stokes.mesh_hierarchy = []
        for nex in [2**i for i in range(1,1+int(np.log2(Nelem[0])))]:
            stokes.NEx_hier.append(nex)
            stokes.NEy_hier.append(nex)
            stokes.mesh_hierarchy.append(RectangleMesh_MyCoord(nex, nex, 2, 2,
                                          quadrilateral=quadrilateral, prescribed_coord=None, shape="L"))
        mesh = stokes.mesh_hierarchy[-1]

    elif mesh_verts is None:
        ref_x   = int(np.log2(Nelem[0]))
        ref_y   = int(np.log2(Nelem[1]))
        ref     = min(ref_x, ref_y)

        for r in range(ref, -1, -1):
            stokes.NEx_hier.append(2 ** (1 + ref_x - r))
            stokes.NEy_hier.append(2 ** (1 + ref_y - r))

        mesh    = RectangleMesh(2 ** (1 + ref_x - ref), 2 ** (1 + ref_y - ref), domain[0], domain[1], quadrilateral=True)
        stokes.mesh_hierarchy = MeshHierarchy(mesh, ref-1, reorder=False).meshes
        mesh    = stokes.mesh_hierarchy[-1]
    else:
        from sysmg.util.firedrake_mesh import RectangleMesh_MyCoord
        stokes.mesh_hierarchy = []
        for verts, nex, ney in zip(mesh_verts, [2 ** i for i in range(1, 1 + int(np.log2(2 * Nelem[0])))],
                            [2 ** i for i in range(int(Nelem[1] / Nelem[0]),
                                                   int(Nelem[1] / Nelem[0]) + int(np.log2(2 * Nelem[0])))]):
            stokes.NEx_hier.append(nex)
            stokes.NEy_hier.append(ney)
            stokes.mesh_hierarchy.append(RectangleMesh_MyCoord(nex, ney, domain[0], domain[1],
                                                                   quadrilateral=quadrilateral, prescribed_coord=verts))
        mesh = stokes.mesh_hierarchy[-1]

    stokes.mesh = mesh


    if printstuff:
        print('end mesh setup')
        print("NE=(%d,%d), Domain: x-dim[0,%.1f] by y-dim[0,%.1f]"
              % (stokes.NEx_hier[-1], stokes.NEy_hier[-1], domain[0], domain[1]))

    if Q1Q1 or isoQ2_Q1:  # PoSD
        stokes.p_fxn_space = lambda m: interpolate(SpatialCoordinate(m), VectorFunctionSpace(m, "CG", 1))
        stokes.v_fxn_space = stokes.p_fxn_space
        stokes.v_order = 1
        stokes.p_order = 1
    else:
        stokes.v_fxn_space = lambda m: interpolate(SpatialCoordinate(m), VectorFunctionSpace(m, "CG", 2))
        stokes.p_fxn_space = lambda m: interpolate(SpatialCoordinate(m), VectorFunctionSpace(m, "CG", 1))
        stokes.v_order = 2
        stokes.p_order = 1

    # get DOF locations
    # useful if we gonna re-order the matrix
    for m in stokes.mesh_hierarchy:
        v_dof_coord = stokes.v_fxn_space(m).dat.data_ro.copy()
        p_dof_coord = stokes.p_fxn_space(m).dat.data_ro.copy()
        stokes.v_dof_coord_hier.append(v_dof_coord)
        stokes.p_dof_coord_hier.append(p_dof_coord)

    # Taylor-Hood discretization
    for m in stokes.mesh_hierarchy:
        V = None; W = None
        if Q1Q1 or isoQ2_Q1:  # PoSD
            V = VectorFunctionSpace(m, "CG", 1)
            W = FunctionSpace(m, "CG", 1)
        else:
            V = VectorFunctionSpace(m, "CG", 2)
            W = FunctionSpace(m, "CG", 1)
        Z = V * W

        stokes.v_space_hier.append(V)
        stokes.p_space_hier.append(W)
        stokes.vp_space_hier.append(Z)

    if printstuff:
        print("VelocitySpace.dim=%d\nPressureSpace.dim=%d\nJoinedSpace.dim=%d"
              % (stokes.v_space_hier[-1].dim(), stokes.p_space_hier[-1].dim(), stokes.vp_space_hier[-1].dim()))

    u, p = TrialFunctions(stokes.vp_space_hier[-1])
    v, q = TestFunctions(stokes.vp_space_hier[-1])

    a = L = None
    f = Constant((0, 0))
    if Q1Q1 or isoQ2_Q1:  # PoSD
        #raise NotImplementedError
        """
        h     = CellSize(mesh)
        beta  = 0.2
        delta = beta*h*h
        a = (inner(grad(u), grad(v)) - p*div(v) - div(u)*q + \
            delta*inner(grad(p), grad(q)))*dx
        L = inner(f, v+delta*grad(q))*dx
        """
        a = (inner(grad(u), grad(v)) - p * div(v) - div(u) * q) * dx
        L = inner(f, v) * dx
    else:
        a = (inner(grad(u), grad(v)) - p * div(v) - div(u) * q) * dx
        L = inner(f, v) * dx

    # BCs
    bcs = []
    if BCs =='bfs':
        for Z in stokes.vp_space_hier:
            (x, y) = SpatialCoordinate(Z.mesh())

            poiseuille_flow = as_vector([1. * (2 - y) * (y - 1) * (y > 1), 0])
            bcs             = [DirichletBC(Z.sub(0), poiseuille_flow, 1),
                               DirichletBC(Z.sub(0), Constant((0., 0.)), (2))]
            stokes.bcs_v_hier.append(bcs)

            bcs_v_nodes = []
            for bc_v in stokes.bcs_v_hier[-1]:
                bcs_v_nodes += list(bc_v.nodes)
            stokes.bcs_v_nodes_hier.append(list(set(bcs_v_nodes)))

        bcs = stokes.bcs_v_hier[-1]
    elif BCs == 'periodic':
        bcs = None
        stokes.bcs_v_nodes_hier = [[] for i in range(len(stokes.vp_space_hier))]
    elif BCs == 'in-out-flow':
        for Z in stokes.vp_space_hier:
            (x, y) = SpatialCoordinate(Z.mesh())

            poiseuille_flow = as_vector([1 - (y - 1) ** 2, 0])
            bcs = [DirichletBC(Z.sub(0), poiseuille_flow, 1),
                    DirichletBC(Z.sub(0), Constant((0., 0.)), (3, 4))]
            stokes.bcs_v_hier.append(bcs)

            bcs_v_nodes = []
            for bc_v in stokes.bcs_v_hier[-1]:
                bcs_v_nodes += list(bc_v.nodes)
            stokes.bcs_v_nodes_hier.append(list(set(bcs_v_nodes)))

        bcs = stokes.bcs_v_hier[-1]
    elif BCs in ['Washer', 'Regularized']:

        for Z, m in zip(stokes.vp_space_hier, stokes.mesh_hierarchy):
            y, x = SpatialCoordinate(m)
            # upside down parabola centered at center
            center = float(domain[0] / 2.)
            ms = domain[0] * domain[1]
            if BCs == 'Regularized':
                fx = 1 - y ** 4  # -(m*(y-center))**2+(center*m)**2
            else:
                fx = -(ms * (y - center)) ** 2 + (center * ms) ** 2

            vec = as_vector([fx, 0])
            bcs_null = [    DirichletBC(Z.sub(0), vec, (4,)),
                            DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]
            stokes.bcs_v_hier.append(bcs_null)

            bcs_v_nodes = []
            for bc_v in stokes.bcs_v_hier[-1]:
                bcs_v_nodes += list(bc_v.nodes)
            stokes.bcs_v_nodes_hier.append(list(set(bcs_v_nodes)))

        bcs = stokes.bcs_v_hier[-1]

    elif BCs == 'zero':
            for Z, m in zip(stokes.vp_space_hier, stokes.mesh_hierarchy):
                bcs_null = [  DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4))]
                stokes.bcs_v_hier.append(bcs_null)

                bcs_v_nodes = []
                for bc_v in stokes.bcs_v_hier[-1]:
                    bcs_v_nodes += list(bc_v.nodes)
                stokes.bcs_v_nodes_hier.append(list(set(bcs_v_nodes)))

            bcs = stokes.bcs_v_hier[-1]
    else:
        BCs = 'None'
        bcs = None
        raise ValueError("BC option not implemented.")

    stokes.bcs_p_nodes_hier = [[] for i in range(len(stokes.vp_space_hier))]

    if printstuff:
        print('BCs type: %s' % BCs)
        print('assembling the system')

    ###########################################################################
    # Form matrices
    A               = assemble(a, bcs=bcs, mat_type='nest', sub_mat_type='aij')
    b               = assemble(L, bcs=bcs)

    Z               = stokes.vp_space_hier[-1]
    nullspace       = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
    nullspace._build_monolithic_basis()
    nullspace_new   = nullspace._nullspace
    petscVec        = nullspace_new.getVecs()[0]
    petscVec.assemble()
    stokes.nullspace = petscVec.getArray()

    # the wrong way to get the right hand-side
    # stokes.b =  np.copy(b.vector().array())
    # the right way to form the right handside
    # read - https://www.firedrakeproject.org/boundary_conditions.html
    #  A x = b - action(A, zero_function_with_bcs_applied)
    # the code below was adopted from linear_solver.py:_lifted.py
    from firedrake.assemble import create_assembly_callable
    from firedrake import ufl_expr
    u = Function(Z)
    blift = Function(Z)

    u.dat.zero()
    for bc in A.bcs:
        bc.apply(u)

    expr = -ufl_expr.action(A.a, u)
    create_assembly_callable(expr, tensor=b)()

    blift += b
    for bc in A.bcs:
        bc.apply(blift)
    stokes.b = np.copy(blift.vector().array())

    # extract subnmatrices
    if printstuff:
        print('extracting submatrices')
    M = A.petscmat.getNestSubMatrix(0, 0)
    BT = A.petscmat.getNestSubMatrix(0, 1)
    B = A.petscmat.getNestSubMatrix(1, 0)
    Zero_block = A.petscmat.getNestSubMatrix(1, 1)

    # There is a minus sign floating around
    # because this is the way matrix is defined in literature
    # | M   B^T |
    # | B   -Z  |
    # Save Submatrices
    stokes.M = sp.csr_matrix((M.getValuesCSR())[::-1])
    stokes.BT = sp.csr_matrix((BT.getValuesCSR())[::-1])
    stokes.B = sp.csr_matrix((B.getValuesCSR())[::-1])
    stokes.Zero_block = -1 * sp.csr_matrix((Zero_block.getValuesCSR())[::-1])

    # obtain  mass matrices
    stokes.Mass_v, stokes.Mass_p = getMassMatrices(stokes, order=(stokes.v_order, stokes.p_order))
    stokes.Pressure_stiff        = getPressureStiffnessMatrix(stokes, order=1)
    # Form Full matrix
    stokes.A_bcsr = sp.bmat([[stokes.M, stokes.BT],
                             [stokes.B, -1 * stokes.Zero_block]])
    stokes.A = stokes.A_bcsr.tocsr()

    # include matrix with  no BCs (for debugging)
    A_no_bcs  = assemble(a, mat_type='nest', sub_mat_type='aij')
    M_no_bcs  = A_no_bcs.petscmat.getNestSubMatrix(0, 0)
    BT_no_bcs = A_no_bcs.petscmat.getNestSubMatrix(0, 1)
    B_no_bcs  = A_no_bcs.petscmat.getNestSubMatrix(1, 0)
    M_no_bcs  = sp.csr_matrix((M_no_bcs.getValuesCSR())[::-1])
    BT_no_bcs = sp.csr_matrix((BT_no_bcs.getValuesCSR())[::-1])
    B_no_bcs  = sp.csr_matrix((B_no_bcs.getValuesCSR())[::-1])
    stokes.A_without_bc  = sp.bmat([[M_no_bcs, BT_no_bcs],
                                    [B_no_bcs, None]]).tocsr()
    stokes.M_without_bc  = M_no_bcs
    stokes.B_without_bc  = B_no_bcs
    stokes.BT_without_bc = BT_no_bcs

    if Solve:
        if printstuff:
            print('solving the system')
        # solve system using firedrake machinery
        upsol = Function(Z)
        usol, psol = upsol.split()

        nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
        solve(a == L, upsol, bcs=bcs, nullspace=nullspace)
        """
              solver_parameters={'pc_type': 'fieldsplit',
                                 'ksp_rtol': 1e-15,
                                 'pc_fieldsplit_type': 'schur',
                                 'fieldsplit_schur_fact_type': 'diag',
                                 'fieldsplit_0_pc_type': 'redundant',
                                 'fieldsplit_0_redundant_pc_type': 'lu',
                                 'fieldsplit_1_pc_type': 'none',
                                 'ksp_monitor_true_residual': None,
                                 'mat_type': 'aij'})
        """
        stokes.upsol_firedrake = upsol

        if Plot:
            fig, axes = plt.subplots()
            triplot(mesh, axes=axes)
            axes.legend()

            if upsol is not None:
                v, p = upsol.split()
                v.rename("Velocity")
                p.rename("Pressure")

                fig, axes = plt.subplots()
                l = tricontourf(v, axes=axes)
                triplot(mesh, axes=axes, interior_kw=dict(alpha=0.05))
                plt.colorbar(l)

                fig, axes = plt.subplots()
                l = tricontourf(p, axes=axes)
                triplot(mesh, axes=axes, interior_kw=dict(alpha=0.05))
                plt.colorbar(l)
                plt.show()

        stokes.upsol = np.copy(upsol.vector().array())
        stokes.usol = np.copy(usol.vector().array())
        stokes.psol = np.copy(psol.vector().array())

    stokes.dof_ordering = dof_ordering
    if dof_ordering['split_by_component']:
        if printstuff:
            print('reorder the system (split_by_component=True): \
                    [x0 y0 .. xn yn] ---> [x0..xn y0 .. yn]')
        stokes_orig = stokes
        stokes = stokes_orig.create_split_v_copy()


    if dof_ordering['renumber_based_on_coord']:
        for v_dof in stokes.v_dof_coord_hier:
            v_dof = v_dof.round(12)
        for p_dof in stokes.p_dof_coord_hier:
            p_dof = p_dof.round(12)
        stokes.renumber_dofs()

    # relative positions of DoFs for GMG. Needs to be called before lo_fe_precond
    # to populate the data structures
    stokes.grid_hierarchy()

    if lo_fe_precond:
        # Form Q1-Q1 system that has the same number of velocity DoFs
        NE_lo = (Nelem[0] * 2, Nelem[1] * 2)

        xmax = domain[0]
        ymax = domain[1]
        if BCs == 'bfs':
            xmax = 2.
            ymax = 2.

        mesh_coord_hier = []
        for nex, ney in zip([2 ** i for i in range(1, 1 + int(np.log2(2 * Nelem[0])))],
                            [2 ** i for i in range(int(Nelem[1] / Nelem[0]), int(Nelem[1] / Nelem[0]) + int(np.log2(2 * Nelem[0])))]):
            xcoords     = np.linspace(0.0, xmax, nex + 1, dtype=np.double)
            ycoords     = np.linspace(0.0, ymax, ney + 1, dtype=np.double)
            mesh_coord_hier.append(np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2))

        lo_fe_sys   = form_stokes(  Nelem=NE_lo, domain=domain,
                                    quadrilateral=quadrilateral,
                                    isoQ2_Q1=True, BCs=BCs,
                                    dof_ordering=dof_ordering,
                                    mesh_verts=mesh_coord_hier)

        # form restriction for pressure DoFs of Q1-Q1 system
        from sysmg.geometric.taylorhood_interp import construct_interp
        Ps = construct_interp(
            lo_fe_sys.p_grid_hier, lo_fe_sys.p_ext_grid_hier,
            [[] for _ in range(len(lo_fe_sys.p_grid_hier))],  # no BCs to impose
            interp_order=1,
            quadrilateral=quadrilateral,
            coarsening_rate=(2, 2),
            periodic=stokes.periodic # need this info for coarsening
        )

        R_2n_to_n   = sp.csr_matrix(Ps[-1]).T
        P_n_to_2n   = sp.csr_matrix(Ps[-1])

        # Restrict Pressure space to a coarser grid to get Q1isoQ2-Q1 system
        M           = lo_fe_sys.M
        B           = R_2n_to_n * lo_fe_sys.B
        BT          = lo_fe_sys.BT * P_n_to_2n
        A_q1q1      = sp.bmat([[M, BT],
                                 [B, None]])

        Mass_p      = R_2n_to_n * lo_fe_sys.Mass_p * R_2n_to_n.T
        Pressure_stiff = R_2n_to_n *lo_fe_sys.Pressure_stiff* R_2n_to_n.T

        lo_fe_sys.A                     = A_q1q1
        lo_fe_sys.B                     = B
        lo_fe_sys.BT                    = BT
        lo_fe_sys.Mass_p                = Mass_p
        lo_fe_sys.Pressure_stiff        = Pressure_stiff
        # lo_fe_sys.Mass_p        = stokes.Mass_p
        lo_fe_sys.NEx_hier              = stokes.NEx_hier
        lo_fe_sys.NEy_hier              = lo_fe_sys.NEy_hier if BCs == 'periodic' else stokes.NEy_hier
        lo_fe_sys.B_without_bc          = R_2n_to_n * lo_fe_sys.B_without_bc
        lo_fe_sys.BT_without_bc         = lo_fe_sys.BT_without_bc * P_n_to_2n
        lo_fe_sys.M_without_bc          = lo_fe_sys.M_without_bc
        lo_fe_sys.Zero_block            = R_2n_to_n * lo_fe_sys.Zero_block * P_n_to_2n
        lo_fe_sys.R_2n_to_n             = R_2n_to_n
        lo_fe_sys.b                     = stokes.b
        lo_fe_sys.periodic              = stokes.periodic
        lo_fe_sys.p_dof_coord_hier      = stokes.p_dof_coord_hier
        lo_fe_sys.p_grid_hier           = stokes.p_grid_hier
        lo_fe_sys.p_ext_grid_hier       = stokes.p_ext_grid_hier

        # get rid of grids not used
        if not lo_fe_sys.periodic:
            lo_fe_sys.v_ext_grid_hier.pop(0)
            lo_fe_sys.v_grid_hier.pop(0)
            lo_fe_sys.bcs_v_nodes_hier.pop(0)
            #            lo_fe_sys.bcs_p_nodes_hier.pop(0)
            lo_fe_sys.v_dof_coord_hier.pop(0)

        if lo_fe_sys.periodic:
            lo_fe_sys.v_grid_hier.pop(0)
            lo_fe_sys.v_ext_grid_hier.pop(0)
            lo_fe_sys.v_dof_coord_hier.pop(0)
            lo_fe_sys.mesh_hierarchy.pop(0)
            lo_fe_sys.NEy_hier.pop(0)
            lo_fe_sys.bcs_v_nodes_hier.pop(0)
            lo_fe_sys.bcs_p_nodes_hier.pop(0)

        assert all([np.allclose(st[0], st[1]) for st in zip(lo_fe_sys.v_ext_grid_hier, stokes.v_ext_grid_hier)])
        assert all([np.allclose(st[0], st[1]) for st in zip(lo_fe_sys.v_grid_hier, stokes.v_grid_hier)])
        assert all([np.allclose(st[0], st[1]) for st in zip(lo_fe_sys.p_ext_grid_hier, stokes.p_ext_grid_hier)])
        assert all([np.allclose(st[0], st[1]) for st in zip(lo_fe_sys.p_grid_hier, stokes.p_grid_hier)])
        assert all([np.allclose(np.sort(st[0]), np.sort(st[1])) for st in zip(lo_fe_sys.bcs_v_nodes_hier, stokes.bcs_v_nodes_hier)])
        assert all([np.allclose(np.sort(st[0]), np.sort(st[1])) for st in zip(lo_fe_sys.bcs_p_nodes_hier, stokes.bcs_p_nodes_hier)])
        assert all([np.allclose(st[0], st[1]) for st in zip(lo_fe_sys.v_dof_coord_hier, stokes.v_dof_coord_hier)])
        assert all([np.allclose(st[0], st[1]) for st in zip(lo_fe_sys.p_dof_coord_hier, stokes.p_dof_coord_hier)])
        """
        """

        stokes.lo_fe_sys                = lo_fe_sys

    if printstuff:
        print('x----------------------------------------------------------------')
    return stokes


def getMassMatrices(stokes, order=(2,1)):
    """
    Construct mass matrices for pressure and velocity fields.

    Args:
        stokes: sysmg.systems.stokes.StokesSystem
            Stokes system
        order: tuple
            order of FE systems.

    Returns:
        tuple : Velocity Mass matrix and Pressure Mass Matrix (csr_matrix).

    """
    # Construct Velocity and Pressure Mass Matrix
    mesh    = stokes.mesh
    V       = VectorFunctionSpace(mesh, "CG", order[0])
    P       =       FunctionSpace(mesh, "CG", order[1])
    Z       = V * P

    V_trial, P_trial = TrialFunctions(Z)
    V_test,  P_test  = TestFunctions(Z)

    mass_v  = inner(V_test, V_trial)*dx
    mass_p  = inner(P_test, P_trial)*dx

    Mass_v  = assemble(mass_v, bcs=[], mat_type='nest', sub_mat_type='aij')
    Mass_p  = assemble(mass_p, mat_type='nest', sub_mat_type='aij')

    Mv      = sp.csr_matrix((Mass_v.petscmat.getNestSubMatrix(0,0).getValuesCSR())[::-1])
    Mp      = sp.csr_matrix((Mass_p.petscmat.getNestSubMatrix(1,1).getValuesCSR())[::-1])

    return Mv, Mp


def getPressureStiffnessMatrix(stokes, order=1):
    """
    Construct pressure stiffness matrix.

    Args:
        stokes: sysmg.systems.stokes.StokesSystem
            Stokes system
        order: tuple
            order of FE system.

    Returns:
        csr_matrix : Pressure stiffness Matrix.

    """
    # Construct Velocity and Pressure Mass Matrix
    mesh    = stokes.mesh
    P       = FunctionSpace(mesh, "CG", order)

    P_trial, P_test = TrialFunction(P), TestFunction(P)

    mass_p  = inner(grad(P_test), grad(P_trial))*dx
    Mass_p  = assemble(mass_p, mat_type='nest', sub_mat_type='aij')
    Mp      = sp.csr_matrix((Mass_p.petscmat.getValuesCSR())[::-1])

    return Mp

if __name__== "__main__":
    stokes = form_stokes(Nelem=(4,4), domain=(2., 2.),
            Solve=True, quadrilateral=True, Q1Q1=False, BCs='Washer')
