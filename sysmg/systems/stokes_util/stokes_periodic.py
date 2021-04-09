from firedrake import *
from firedrake.mg.mesh import HierarchyBase
import numpy

# This code was borrowed from Patrick E. Farrell's repo, which I have trouble finding now.

distribution_parameters={"partition":True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

class ApproximateSchur(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        Z = test.function_space()
        a = - inner(test, trial)*dx
        bcs = []
        return (a, bcs)



class PeriodicProblem(object):
    def __init__(self, baseN, nref, use_quads=True):
        super().__init__()
        self.baseN     = baseN
        self.nref      = nref
        self.use_quads = use_quads

    def periodise(self, m):
        # stolen from  firedrake/src/firedrake/firedrake/utility_meshes.py
        family          = 'DQ'            if self.use_quads else 'DG'
        cell            = 'quadrilateral' if self.use_quads else 'triangle'
        coord_fs        = VectorFunctionSpace(m, FiniteElement(family, cell, 1, variant="equispaced"), dim=2)
        old_coordinates = m.coordinates
        new_coordinates = Function(coord_fs)

        domain = "{[i, j]: 0 <= i < old_coords.dofs and 0 <= j < new_coords.dofs}"
        instructions = """
        <float64> pi = 3.141592653589793
        <float64> eps = 1e-12
        <float64> bigeps = 1e-1
        <float64> Y = 0
        <float64> Z = 0
        for i
            Y = Y + old_coords[i, 1]
            Z = Z + old_coords[i, 2]
        end
        for j
            <float64> phi = atan2(old_coords[j, 1], old_coords[j, 0])
            <float64> _phi = abs(sin(phi))
            <double> _theta_1 = atan2(old_coords[j, 2], old_coords[j, 1] / sin(phi) - 1)
            <double> _theta_2 = atan2(old_coords[j, 2], old_coords[j, 0] / cos(phi) - 1)
            <float64> theta = _theta_1 if _phi > bigeps else _theta_2
            new_coords[j, 0] = phi / (2 * pi)
            new_coords[j, 0] = new_coords[j, 0] + 1 if new_coords[j, 0] < -eps else new_coords[j, 0]
            <float64> _nc_abs = abs(new_coords[j, 0])
            new_coords[j, 0] = 1 if _nc_abs < eps and Y < 0 else new_coords[j, 0]
            new_coords[j, 1] = theta / (2 * pi)
            new_coords[j, 1] = new_coords[j, 1] + 1 if new_coords[j, 1] < -eps else new_coords[j, 1]
            _nc_abs = abs(new_coords[j, 1])
            new_coords[j, 1] = 1 if _nc_abs < eps and Z < 0 else new_coords[j, 1]
            new_coords[j, 0] = new_coords[j, 0] * Lx[0]
            new_coords[j, 1] = new_coords[j, 1] * Ly[0]
        end
        """

        cLx = Constant(1)
        cLy = Constant(1)

        par_loop((domain, instructions), dx,
                 {"new_coords": (new_coordinates, WRITE),
                  "old_coords": (old_coordinates, READ),
                  "Lx": (cLx, READ),
                  "Ly": (cLy, READ)},
                 is_loopy_kernel=True)

        return Mesh(new_coordinates)

    @staticmethod
    def snap(mesh, N, L=1):
        coords = mesh.coordinates.dat.data
        coords[...] = numpy.round((N/L)*coords)*(L/N)

    def mesh(self):
        base = TorusMesh(self.baseN, self.baseN, 1.0, 0.5,
                             distribution_parameters=distribution_parameters, quadrilateral=self.use_quads)
        mh = MeshHierarchy(base, self.nref, distribution_parameters=distribution_parameters)

        meshes = tuple(self.periodise(m) for m in mh)
        mh = HierarchyBase(meshes, mh.coarse_to_fine_cells, mh.fine_to_coarse_cells)

        for (i, m) in enumerate(mh):
            if i > 0:
                self.snap(m, self.baseN * 2**i)

        # Distort coordinates?
        eps = 0.0 # Measure of distortion, eps = 0 is no distortion
        (x, y) = SpatialCoordinate(mh[0])
        V = FunctionSpace(mh[0], mh[0].coordinates.ufl_element())
        new = Function(V).interpolate(as_vector([x + eps*sin(2*pi*x)*sin(2*pi*y),
                                                 y - eps*sin(2*pi*x)*sin(2*pi*y)]))
        coords = [new]
        for mesh in mh[1:]:
            fine = Function(mesh.coordinates.function_space())
            prolong(new, fine)
            coords.append(fine)
            new = fine
        for (mesh, coord) in zip(mh, coords):
            mesh.coordinates.assign(coord)

        # Save the finest mesh to disk for visualisation
        V = FunctionSpace(mh[-1], mh[-1].coordinates.ufl_element())
        u = Function(V)
        File("/tmp/eps-%s.pvd" % eps).write(u)

        return mh[-1]

    def bcs(self, Z):
        return None

    def has_nullspace(self): return True

    def analytical_solution(self, mesh):
        (x, y) = SpatialCoordinate(mesh)
        u_exact = as_vector([cos(2*pi*y), sin(2*pi*x)])
        return u_exact

    def src(self, mesh):
        u_exact = self.analytical_solution(mesh)
        f = -div(grad(u_exact))
        return f
