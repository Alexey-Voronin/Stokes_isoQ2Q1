import numpy as np
import ufl
from pyop2.mpi import COMM_WORLD
from pyop2.datatypes import IntType

from firedrake import VectorFunctionSpace, Function, Constant, \
par_loop, dx, WRITE, READ, interpolate, FiniteElement, interval, tetrahedron
from firedrake.cython  import dmcommon
from firedrake import mesh
from firedrake import function
from firedrake import functionspace

# Added argument to to specify vertex locations prescribed_coord needs to be lex-sorted.
# Warning: If the mesh connectivity comes out funky, it's because the coordinates
# in each col/ may not have exactly the same x coordinate (round-off error)
# Fix: just round all coardinates to some reasonable value.

def RectangleMesh_MyCoord(nx, ny, Lx, Ly, quadrilateral=False, reorder=None, shape='rectangle',
                  diagonal="left", distribution_parameters=None, comm=COMM_WORLD, prescribed_coord=None):
    """Generate a rectangular mesh (adapted from firedrake package).

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    :kwarg diagonal: For triangular meshes, should the diagonal got
        from bottom left to top right (``"right"``), or top left to
        bottom right (``"left"``), or put in both diagonals (``"crossed"``).

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == Lx
    * 3: plane y == 0
    * 4: plane y == Ly
    """
    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(0.0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0.0, Ly, ny + 1, dtype=np.double)
    coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
    # cell vertices
    i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))
    if not quadrilateral and diagonal == "crossed":
        dx = Lx * 0.5 / nx
        dy = Ly * 0.5 / ny
        xs = np.linspace(dx, Lx - dx, nx, dtype=np.double)
        ys = np.linspace(dy, Ly - dy, ny, dtype=np.double)
        extra = np.asarray(np.meshgrid(xs, ys)).swapaxes(0, 2).reshape(-1, 2)
        coords = np.vstack([coords, extra])
        #
        # 2-----3
        # | \ / |
        # |  4  |
        # | / \ |
        # 0-----1
        cells = [i*(ny+1) + j,
                 i*(ny+1) + j+1,
                 (i+1)*(ny+1) + j,
                 (i+1)*(ny+1) + j+1,
                 (nx+1)*(ny+1) + i*ny + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 5)
        idx = [0, 1, 4, 0, 2, 4, 2, 3, 4, 3, 1, 4]
        cells = cells[:, idx].reshape(-1, 3)


    else:

        cells = [i*(ny+1) + j, i*(ny+1) + j+1, (i+1)*(ny+1) + j+1, (i+1)*(ny+1) + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
        if not quadrilateral:
            if diagonal == "left":
                idx = [0, 1, 3, 1, 2, 3]
            elif diagonal == "right":
                idx = [0, 1, 2, 0, 2, 3]
            else:
                raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
            # two cells per cell above...
            cells = cells[:, idx].reshape(-1, 3)

        if shape == "L" :
            keep_nodes = [ ]
            rm_nodes   = [ ]
            for i in range(coords.shape[0]):
                if coords[i,0] >= 1 and coords[i,1] >= 1 or coords[i,0] >= 1 or coords[i,1] >= 1:
                    keep_nodes.append(i)
                else:
                    rm_nodes.append(i)
            """
            print('rm_nodes:\n', rm_nodes)
            plt.scatter(coords[:,0], coords[:,1]);
            plt.scatter(coords[keep_nodes,0], coords[keep_nodes,1], marker='x');
            plt.show()
            """

            rm_nodes = np.array(rm_nodes)
            keep_cells = []
            for c in cells:
                skip = False
                for node in c:
                    if node  in rm_nodes:
                        skip = True
                        break
                if not skip:
                    keep_cells.append(c)

            c = 0
            dof_map = dict()
            for node in keep_nodes:
                dof_map[node] = c
                c +=1

            cells  = np.array(keep_cells)
            coords = coords[keep_nodes,:]
            for c in cells:
                for i in range(len(c)):
                    c[i] = dof_map[c[i]]


        #    print('after:')
        #    print(coords.shape)
        #    print(cells.shape)

    """
    import matplotlib.pyplot as plt
    plt.scatter(coords[:,0], coords[:,1])
    plt.show()
    print('cells:\n', cells)
    print('coords:\n', coords)
    """
    if prescribed_coord is None:
        plex = mesh._from_cell_list(2, cells, coords, comm)
    else:
        plex = mesh._from_cell_list(2, cells, prescribed_coord, comm)

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords    = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)

        if shape == "L" :
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if abs(face_coords[0]) < xtol and abs(face_coords[2]) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
                if abs(face_coords[0] - Lx) < xtol and abs(face_coords[2] - Lx) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
                if abs(face_coords[0] - 1) < xtol and abs(face_coords[2] - 1) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)

                if abs(face_coords[1]) < ytol and abs(face_coords[3]) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
                if abs(face_coords[1] - Ly) < ytol and abs(face_coords[3] - Ly) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
                if abs(face_coords[1] - 1) < ytol and abs(face_coords[3] - 1) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
        else:
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if abs(face_coords[0]) < xtol and abs(face_coords[2]) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
                if abs(face_coords[0] - Lx) < xtol and abs(face_coords[2] - Lx) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
                if abs(face_coords[1]) < ytol and abs(face_coords[3]) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
                if abs(face_coords[1] - Ly) < ytol and abs(face_coords[3] - Ly) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 4)

    return mesh.Mesh(plex, reorder=reorder, distribution_parameters=distribution_parameters)
