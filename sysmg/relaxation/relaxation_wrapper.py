# allows one to access/change cycle-type and other relaxation
# object parameters after creating an ml object
class RelaxationWrapper(object):
    """
    Relaxation object wrapper.

    Relaxation object wrapper that allows one to access/change relaxation
    specific parameters. Created to improve setup speeds and minimize the
    codes changes in pyamg module.
    """

    relax_obj = None
    relax_fxn = None
    def __init__(self, relax_obj, relax_fxn):
        """
        Object initialization.

        Args:
            relax_obj: (sysmg.relxation.vanka, sysmg.relxation.braesssarazin)
                Relaxation object.
            relax_fxn: (function)
                Function called in pyamg to perform relaxation.

        """
        self.relax_obj  = relax_obj
        self.relax_fxn  = relax_fxn
    def __call__(self, A, x, b, *args):
        """
        Call to relaxation.

        Args:
            A: (csr_matrix)
                Systems matrix (ignored, but needed to be compatible with pyamg
                relaxation).

            x: (numpy.ndarray)
                Initial guess.

            b: (numpy.ndarray)
                Right hand side.

            args: (dict)
                ignored.

        Returns:
            None, x will be modified in place.

        Notes:
            If you would like to chain parameters then modify the relaxation
            object via the helper functions.

        """
        self.relax_fxn(A, x, b, *args)
