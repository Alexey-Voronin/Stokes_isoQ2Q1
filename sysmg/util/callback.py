import numpy as np

class CallBack_r(object):
    """
    Callback function that taken residual value as an input.

    Primarily used with SciPy's implementation of GMRES.

    """

    resid        = None
    def __init__(self):
        self.resid        = []
    def __call__(self, inp=None):
        self.resid.append(inp)

class CallBack_x(object):
    """
    Callback function that takes appproximate solution as an input.

    It is used to compute and record residual history.

    """
    
    resid        = None
    A            = None
    b            = None
    def __init__(self, A, b, M=None):
        self.resid        = []
        self.A            = A
        self.b            = b
        self.M            = M
    def __call__(self, x=None):
        r  = self.b-self.A*x
        if self.M is None:
            self.resid.append(np.linalg.norm(r))
        else:
            self.resid.append(np.linalg.norm(self.M*r))
