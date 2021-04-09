from firedrake import *

import numpy as np
import scipy
import scipy.sparse as sp
import pyamg

# system setup
from .systems.stokes        import *
# Multigrid constructors
from .h_multigrid           import MG, make_precond
from .ph_multigrid          import iso_mg
# Solver
from .util.solver           import solve
# Visualization
from .systems.stokes_util.stokes_plots import plot_guess
