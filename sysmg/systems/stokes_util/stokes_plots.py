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

        #upsol_list[i][:] =  stokes.P_split.T*(stokes.P_renumber.T*upsol_list[i])

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
    #sol_renumbered = stokes.P_split.T*(stokes.P_renumber.T*stokes.upsol)
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
#        if plot == 'error':
#            div_uguess.vector().dat.data[:] -= project(div(vexact), stokes.p_space).vector().dat.data[:]
#            div_uguess.vector().dat.data[:] = np.abs(div_uguess.vector().dat.data[:] )

        udiv_tmp = div_uguess.vector().array()
        udivmin  = min(udivmin, min(udiv_tmp))
        udivmax  = max(udivmax, max(udiv_tmp))
        ##################################################
        # grad Pressure
        grad_pguess = project(grad(pguess), stokes.v_space_hier[-1])
        if plot == 'error':
            grad_pguess.vector().dat.data[:,:] -= project(grad(pexact), stokes.v_space_hier[-1]).vector().dat.data[:, :]
#            grad_pguess.vector().dat.data[:,:] = np.abs(grad_pguess.vector().dat.data[:,:])
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


            tresid = resids['up'] #np.array(tresid)/tresid[0];
            uresid = resids['u'] #np.array(uresid)/uresid[0];
            presid = resids['p'] #np.array(presid)/presid[0];
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
        l  = tricontourf(div_uguess, axes=ax) #,norm=colors.SymLogNorm(linthresh=1e-8, vmin=udivmin, vmax=udivmax), vmin=udivmin, vmax=udivmax)
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
        l  = tricontourf(p_grad_guess, axes=ax) #, norm=colors.SymLogNorm(linthresh=1e-6, vmin=pgradmin, vmax=pgradmax)) #, locator=ticker.LogLocator(), vmin=1e-8) #, vmin=pmin, vmax=pmax)
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


    #if prefix is not None:
    #    os.system("convert %s*.pdf %s.gif" % (prefix, prefix))
    #    os.system("rm %s-*.pdf" % prefix)

def plot_fields_minimal(stokes, psol, vsol):
    ##################################
    #Plot velocity
    nx     = stokes.NEx*2+1
    ny     = stokes.NEy*2+1

    assert len(vsol) == nx*ny*2, "velocity dofs don't match"

    x      = np.linspace(0, stokes.domain[1], nx)
    y      = np.linspace(0, stokes.domain[0], ny)
    dx     = x[1]-x[0]
    dy     = x[1]-x[0]
    X, Y   = np.meshgrid(x, y)


    Vsol        = np.zeros_like(X)
    v_dof_coord = stokes.v_dof_coord
    for i in range(v_dof_coord.shape[0]):
        Vsol[int(v_dof_coord[i][0]/dx), int(v_dof_coord[i][1]/dy)] = vsol[i]

    plt.figure(figsize=(12,3))
    plt.subplot(121)
    ax = plt.gca()
    ax.contour(Y, X, Vsol, 20, cmap='RdGy')

    nx     = stokes.NEx+1
    ny     = stokes.NEy+1

    assert len(psol) == nx*ny, "pressure dofs don't match"

    x      = np.linspace(0, stokes.domain[1], nx)
    y      = np.linspace(0, stokes.domain[0], ny)
    dx     = x[1]-x[0]
    dy     = x[1]-x[0]
    X, Y   = np.meshgrid(x, y)


    Psol        = np.zeros_like(X)
    p_dof_coord = stokes.p_dof_coord
    for i in range(p_dof_coord.shape[0]):
        Psol[int(p_dof_coord[i][0]/dx), int(p_dof_coord[i][1]/dy)] = psol[i]


    plt.subplot(122)
    ax = plt.gca()
    ax.contour(Y, X, Psol,  20, cmap='RdGy')
    plt.show()

###############################################################
# Plots one SA aggregate per plot
def plot_aggs_on_dofs(ml, stokes):
    vdofs  = stokes.v_dof_coord.T
    pdofs  = stokes.p_dof_coord.T

    AggOp  = ml.levels[0].AggOp.todense()
    v_len  = int(stokes.M.shape[0])
    vx_len = int(stokes.M.shape[0]/2)
    if stokes.dof_ordering['split_by_component']: #stokes.nodalDofdof_ordering:
        AggOp_vx = AggOp[0:v_len:2]
        AggOp_vy = AggOp[1:v_len:2]
    else:
        AggOp_vx = AggOp[0:vx_len]
        AggOp_vy = AggOp[vx_len:2*vx_len]

    AggOp_p = AggOp[v_len:]

    AGGs   = [AggOp_vx, AggOp_vy, AggOp_p]
    labels = ['Agg_vx_dofs', 'Agg_vy_dofs', 'Agg_p_dofs']
    DOFs   = [vdofs, vdofs, pdofs]

    for i in range(len(AGGs)):
        agg_comp = AGGs[i]
        name     = labels[i]
        dofs     = DOFs[i]
        for idx in range(agg_comp.shape[1]):
            agg = [i for i, e in enumerate(agg_comp[:,idx]) if abs(e) > 0]


            plt.scatter(pdofs[0], pdofs[1], s=7, label="pressure dofs")
            plt.scatter(vdofs[0], vdofs[1], s=2, label="velocity dofs")
            plt.scatter(dofs[0][agg], dofs[1][agg], alpha=0.2, label='aggregate')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('%s: AggOp[%d]' % (name, idx))
            plt.savefig(("%s_p{0:03d}" % name).format(idx), dpi=200, bbox_inches='tight')
            plt.close()
###########################################################################
# highlight on the scatter dof plot matrix entries that correspond to
# dirichlet BCs
def plot_indep_nodes(stokes):
    nnz_per_row   = np.count_nonzero(stokes.A.todense(), axis=1)
    nv            = stokes.M.shape[0]
    Mdofs = np.where(nnz_per_row[:nv].flatten() == 1)[1]
    Bdofs = np.where(nnz_per_row[nv:].flatten() == 1)[1]

    vdofs = stokes.v_dof_coord
    pdofs = stokes.p_dof_coord
    plt.scatter(vdofs[:,0], vdofs[:,1], label='all dofs')

    nvx   = len(Mdofs)//2
    Mdofs = Mdofs[:nvx]
    plt.scatter(vdofs[Mdofs,0], vdofs[Mdofs,1], s=80, alpha=0.3, label='velocity BCs')
    plt.scatter(pdofs[Bdofs,0], pdofs[Bdofs,1], marker='x', s=100, label='pressure BCs')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
