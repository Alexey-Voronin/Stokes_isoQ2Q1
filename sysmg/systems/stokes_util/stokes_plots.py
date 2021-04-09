from firedrake import *
# solvers and data structures
import numpy as np
import scipy.sparse as sp
# plotting
import matplotlib.pyplot as plt
import os

# copies the solution vector back into firedrake function object
# and plots the velocity pressure and divergence of velocity side by side
def plot_guess(stokes, up_approx, scale=True, plot='solution', resids=None, skip=None, prefix=None, title=''):
    """
    Plots thei velocity, pressure, and divergence of velocity side by side.

    Since the stokes system and up_approx has different ordering than the
    firedrake's system, solution vector is copied back into firedrake
    function object.
    """
    if prefix is not None:
        os.system("rm %s*.pdf" % prefix)

    upsol_list = up_approx
    if not isinstance(up_approx, list):
        upsol_list = [up_approx.copy()]
    else:
        upsol_list = up_approx.copy()

    for i in  range(len(upsol_list)):
        upsol_list[i][:] = upsol_list[i]
        if stokes.P_renumber is not None:
            upsol_list[i][:] = stokes.P_renumber.T*upsol_list[i][:]
        if stokes.P_split is not None:
            upsol_list[i][:] = stokes.P_split.T   *upsol_list[i][:]

    umin = umax = pmin = pmax = udivmin = udivmax = None

    maxVal  = minVal = None
    maxVal  = np.finfo(float).max
    minVal  = np.finfo(float).min
    pgradmin= maxVal;  pgradmax=minVal
    udivmin = maxVal;  udivmax=minVal
    if scale:
        umin   = maxVal;  umax   = minVal
        pmin   = maxVal;  pmax   = minVal

    u_sols = []
    p_sols = []
    u_divs = []
    p_grad = []

    tresid = []; uresid = []; presid  = []

    sol_renumbered = None
    if plot == 'error':
        sol_renumbered = stokes.upsol.copy()
        if stokes.P_renumber is not None:
            sol_renumbered = stokes.P_renumber.T*sol_renumbered
        if stokes.P_split is not None:
            sol_renumbered = stokes.P_split.T*sol_renumbered
    # construct firedrake objection to plot
    # regord min/max values for plotting
    for up_sol in upsol_list:

        if plot == 'error':
            up_approx = up_sol - sol_renumbered
        else:
            up_approx = up_sol

        # object used to plot the result
        up_empty       = Function(stokes.vp_space_hier[-1])
        uguess, pguess = up_empty.split()
        # vectors containing the result data
        u_sol          = up_approx[:stokes.M.shape[0]]
        p_sol          = up_approx[stokes.M.shape[0]:] # plot solution

        resid          = stokes.b-stokes.A*up_approx
        tresid.append(np.linalg.norm(resid))
        uresid.append(np.linalg.norm(resid[:stokes.M.shape[0]]))
        presid.append(np.linalg.norm(resid[stokes.M.shape[0]:]))

        ##################################################
        # Velocity
        # move the result data into plotting object
        usol_renumbered               = u_sol.reshape((stokes.M.shape[0]//2, 2))
        #usol_renumbered               = stokes.Pv.T*u_sol.reshape((stokes.M.shape[0]//2, 2))
        uguess.vector().dat.data[:,:] = usol_renumbered
        ##################################################
        # Pressure
        pguess.vector().dat.data[:]   = up_sol[stokes.M.shape[0]:] #p_sol
        ##################################################
        # divergence of Velocity
        if plot == 'error':
            vexact,pexact = stokes.upsol_tmp.split()
        div_uguess = project(div(uguess), stokes.p_space_hier[-1])

        udiv_tmp = div_uguess.vector().array()
        udivmin  = min(udivmin, min(udiv_tmp))
        udivmax  = max(udivmax, max(udiv_tmp))
        ##################################################
        # grad Pressure
        grad_pguess = project(grad(pguess), stokes.v_space_hier[-1])
        if plot == 'error':
            grad_pguess.vector().dat.data[:,:] -= project(grad(pexact), stokes.v_space_hier[-1]).vector().dat.data[:, :]
        pgradmin = min(pgradmin, np.min(grad_pguess.vector().dat.data[:,:]))
        pgradmax = max(pgradmax, np.max(grad_pguess.vector().dat.data[:,:]))

        u_sols.append(uguess)
        p_sols.append(pguess)
        u_divs.append(div_uguess)
        p_grad.append(grad_pguess)

        if scale:
            umin     = min(umin, min(u_sol)); umax = max(umax, max(u_sol))
            pmin     = min(pmin, min(p_sol)); pmax = max(pmax, max(p_sol))

    # plot firerake functions
    count = 0
    for uguess, pguess, div_uguess, p_grad_guess in zip(u_sols, p_sols, u_divs, p_grad):
        plt.figure(figsize=(30,3))

        offset = 0
        if skip is not None:
            offset += 11
            plt.subplot(151)


            tresid = resids['up']
            uresid = resids['u']
            presid = resids['p']
            iters = np.linspace(0, len(tresid)-1, len(tresid), dtype=int)

            plt.semilogy(iters, tresid, marker='o', label='total')
            plt.semilogy(iters, uresid, marker='x', label='momentum')
            plt.semilogy(iters, presid, label='incompressibility')

            plt.title(r'residual%s' % title); plt.xlabel('iteration'); plt.ylabel('residual')
            plt.legend()
            plt.axvline(x=count, c='r')
            count+= skip

        ##################################################
        # Velocity
        plt.subplot(141+offset)
        ax = plt.gca()
        triplot(stokes.mesh, axes=ax, interior_kw=dict(alpha=0.05))
        l = tricontourf(uguess, axes=ax, vmin=umin, vmax=umax)
        p = plt.colorbar(l)
        p.ax.tick_params(labelsize=18)
        plt.title(r"velocity %s, $u_{%s}$" % (plot, plot),  fontsize=20)

        ##################################################
        # divergence of Velocity
        plt.subplot(142+offset)
        ax = plt.gca()
        l  = tricontourf(div_uguess, axes=ax)
        triplot(stokes.mesh, axes=ax, interior_kw=dict(alpha=0.05))
        p = plt.colorbar(l)
        p.ax.tick_params(labelsize=18)
        plt.title(r"%s $\nabla\cdot u$" % plot,  fontsize=20)

        ##################################################
        # Pressure
        plt.subplot(143+offset)
        ax = plt.gca()
        l  = tricontourf(pguess, axes=ax) #, vmin=pmin, vmax=pmax)
        triplot(stokes.mesh, axes=ax, interior_kw=dict(alpha=0.05))
        p = plt.colorbar(l)
        p.ax.tick_params(labelsize=18)
        plt.title(r"pressure solution, $p$", fontsize=20)

        ##################################################
        # grad pressure
        plt.subplot(144+offset)
        ax = plt.gca()
        l  = tricontourf(p_grad_guess, axes=ax)
        triplot(stokes.mesh, axes=ax, interior_kw=dict(alpha=0.05))
        p = plt.colorbar(l)
        p.ax.tick_params(labelsize='large')
        plt.title(r"%s$ \nabla p$" % plot, fontsize=20)


        if prefix is not None:
            num = str(count)
            num = num.zfill(3)
            new_filename = prefix +"-"+ num + ".pdf"
            plt.savefig(new_filename)
        plt.show()

