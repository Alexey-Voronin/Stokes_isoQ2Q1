from firedrake import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


def getInterpStencils(order = 1, quadrilateral=True, DEBUG=False):
    """Compute Q1 and Q1 intetrpolation operator stencil for each DoF type."""
    all_stencils = dict()

    if order == 1:
        mesh       = UnitSquareMesh(2, 2, quadrilateral=quadrilateral)
        # extract coordinates of
        coord_obj  = SpatialCoordinate(mesh)
        coord_data = interpolate(coord_obj, VectorFunctionSpace(mesh, "CG", 2))
        coord      = coord_data.dat.data_ro
        # location of DOFs
        Xs         = coord.T[0];
        dx         = min(Xs[Xs.nonzero()])
        loc_idx    = (coord.T/dx).astype(int)

        # define function space over coarse mesh
        V      = FunctionSpace(mesh, "CG", 1)
        arr    = np.zeros((V.dim()));
        # define FE element function with center DOF set to 1
        if quadrilateral: #discritization == 'Q2-Q1':
            arr[2] = 1 # need to serach for that center DOF
        else: # P2-P1
            arr[3] = 1 # need to serach for that center DOF
        y      = Function(V, np.copy(arr))

        if DEBUG:
            fig, axes = plt.subplots()
            triplot(mesh, axes=axes)
            axes.legend()
            tricontourf(y);
            plt.title('coarse mesh with u=1 @ center DOF'); plt.show()
            fig.show()

        interp_stencil_p1 = np.zeros((5, 5))
        mesh_fine         = UnitSquareMesh(2, 2, quadrilateral=quadrilateral)
        if DEBUG:
            fig, axes = plt.subplots()
            triplot(mesh_fine, axes=axes);
            plt.title('fine mesh with coarse FE function evaluated on fine DOFS ')
            plt.scatter(coord.T[0], coord.T[1], color='r')

        for p in coord.tolist():
            val = y.at(p[0], p[1])
            if abs(val) > 1e-8:
                interp_stencil_p1[int((1.-p[1])/dx),int((p[0])/dx)] = val

                if DEBUG:
                    plt.annotate(("%.0f" % (8*val)), (p[0], p[1]+0.02), fontsize=10)

        interp_stencil_p1 = interp_stencil_p1[1:-1,1:-1].T

        if DEBUG:
            plt.show()
            print(interp_stencil_p1)
            plot(mesh_fine); plt.title('DOF indecies')
            plt.scatter(coord.T[0], coord.T[1], color='r')
            for p in coord.tolist():
                plt.annotate(("(%d, %d)" % (int((1.-p[1])/dx),int((p[0])/dx))),
                               (p[0], p[1]+0.02), fontsize=10)
            plt.show()

        all_stencils['nodal'] = interp_stencil_p1

    elif order == 2:
        mesh_fine  = UnitSquareMesh(4, 4, quadrilateral=quadrilateral)
        coord_obj  = SpatialCoordinate(mesh_fine)
        coord_data = interpolate(coord_obj, VectorFunctionSpace(mesh_fine, "CG", 2))
        coord      = coord_data.dat.data_ro

        mesh_coarse = UnitSquareMesh(2, 2, quadrilateral=quadrilateral)
        if DEBUG:
            fig, axes = plt.subplots()
            triplot(mesh_coarse, axes=axes);
            plt.show()
        V = FunctionSpace(mesh_coarse, "CG", 2)

        dofs_per_dim              = int(np.sqrt((len(coord.T[0]))))
        interp_stencil_p2_node    = np.zeros((dofs_per_dim, dofs_per_dim))
        interp_stencil_p2_vside   = interp_stencil_p2_node.copy()
        interp_stencil_p2_hside   = interp_stencil_p2_node.copy()
        interp_stencil_p2_center  = interp_stencil_p2_node.copy()
        stencils                  = [interp_stencil_p2_node, interp_stencil_p2_vside,
                                     interp_stencil_p2_hside, interp_stencil_p2_center]
        dx                        = min(coord[0][coord[0].nonzero()])

        dof_ids = None
        if quadrilateral: #discritization == 'Q2-Q1':
            dof_selected = [7, 18, 2, 0]
        else:
            dof_selected = [8, 6, 9]

        for name, idx, stencil in zip(['node dof', 'vside dof', 'hside dof', 'center dof'], dof_selected, stencils):
            arr      = np.zeros((V.dim()));
            arr[idx] = 1 # need to search for that center DOF
            y        = Function(V, np.copy(arr))

            if DEBUG:
                plot(mesh_fine); plt.title(name)
                plt.scatter(coord.T[0], coord.T[1], color='r')

            for p in coord.tolist():
                val = y.at(p[0], p[1])
                if abs(val) > 1e-8:
                    stencil[int((1.-p[1])/dx),int((p[0])/dx)] = val
                    if DEBUG:
                        plt.annotate(("%.0f" % val), (p[0], p[1]+0.02), fontsize=10)
            if DEBUG:
                plt.show()


        if quadrilateral: #discritization == 'Q2-Q1':
            interp_stencil_p2_node   = stencils[0][1:-1, 1:-1]
            interp_stencil_p2_vside  = stencils[1][1:4, 1:8]
            interp_stencil_p2_hside  = stencils[2][1:8, 1:4]
            interp_stencil_p2_center = stencils[3][5:8, 1:4]
        else:
            interp_stencil_p2_node   = stencils[0][1:-1, 1:-1]
            interp_stencil_p2_hside  = stencils[1][2:-2, 1:-5]
            interp_stencil_p2_vside  = stencils[1][2:-2, 1:-5].T
            interp_stencil_p2_center = stencils[2][5:-1, 5:-1]

        all_stencils['nodal']      = interp_stencil_p2_node
        all_stencils['vertical']   = interp_stencil_p2_vside
        all_stencils['horizontal'] = interp_stencil_p2_hside
        all_stencils['center']     = interp_stencil_p2_center

        if DEBUG:
            print('interp_stencil_p2_node\n', 8*interp_stencil_p2_node)
            print('sum(stencil)=', sum(interp_stencil_p2_node.flatten()))
            #
            print('interp_stencil_p2_hside\n', 8*interp_stencil_p2_hside)
            print('sum(stencil)=', sum(interp_stencil_p2_hside.flatten()))
            #
            print('interp_stencil_p2_center\n', 8*interp_stencil_p2_center)
            print('sum(stencil)=', sum(interp_stencil_p2_center.flatten()))
            #
            print('interp_stencil_p2_vside\n', 8*interp_stencil_p2_vside)
            print('sum(stencil)=', sum(interp_stencil_p2_vside.flatten()))
    else:
        raise ValueError('Error in getInterpStencils: order=%d not implemented' % order)


    return all_stencils


if __name__== "__main__":
    """
    stencils_p1 = getInterpStencils(order=1, discritization='P2-P1', DEBUG=False)
    stencils_p2 = getInterpStencils(order=2, discritization='P2-P1', DEBUG=False)
    print('P1 Stencils-------------------------------------------')
    for name in stencils_p1.keys():
        print(name); print(stencils_p1[name])
    print('P2 Stencils-------------------------------------------')
    for name in stencils_p2.keys():
        print(name); print(stencils_p2[name])
    print(stencils_q1)
    print(stencils_p1)
    """
    stencils_q1 = getInterpStencils(order=1, quadrilateral=False, DEBUG=False)
    stencils_q2 = getInterpStencils(order=2, quadrilateral=False, DEBUG=False)
    print('Q1 Stencils-------------------------------------------')
    for name in stencils_q1.keys():
        print(name); print(stencils_q1[name])
    print('Q2 Stencils-------------------------------------------')
    for name in stencils_q2.keys():
        print(name); print(stencils_q2[name])
