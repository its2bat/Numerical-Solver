"""3D Transient FEM solver with convergence auto-stop."""
import numpy as np
import time as timer
from scipy.sparse.linalg import factorized
from .params import *
from .fem_core import (assemble_KM_tet, apply_robin_3d,
                       build_sparse, mirror_3d)


def solve_transient_3d(mesh: MeshResult, mat: MaterialParams,
                       bc: BCParams, geom: GeometryParams,
                       sp: SolverParams,
                       log=None, progress=None) -> SolveResult:
    P = mesh.coords
    Nn = mesh.n_nodes

    f = np.zeros(Nn)
    K_rows, K_cols, K_vals = [], [], []
    M_rows, M_cols, M_vals = [], [], []

    for tets, k, rho, cp, name in [
        (mesh.tets_tube, mat.k_tube, mat.rho_tube, mat.cp_tube, "tube"),
        (mesh.tets_paste, mat.k_paste, mat.rho_paste, mat.cp_paste, "paste"),
        (mesh.tets_mold, mat.k_solder, mat.rho_solder, mat.cp_solder, "mold"),
    ]:
        kr, kc, kv, mr, mc, mv = assemble_KM_tet(tets, P, k, rho, cp)
        K_rows.append(kr); K_cols.append(kc); K_vals.append(kv)
        M_rows.append(mr); M_cols.append(mc); M_vals.append(mv)
        if log:
            log(f"  assembled {name}: {len(tets)} tets")

    # Robin BC
    rob_r, rob_c, rob_v, f_rob = apply_robin_3d(
        mesh.tri_rob, P, bc.h_conv, bc.Tinf)
    K_rows.append(rob_r); K_cols.append(rob_c); K_vals.append(rob_v)
    f += f_rob

    K_full = build_sparse(K_rows, K_cols, K_vals, Nn)
    M_full = build_sparse(M_rows, M_cols, M_vals, Nn)

    # Neumann flux on inner cylinder faces (if requested)
    if bc.bc_inner == "neumann":
        a, b, c = mesh.tri_dir[:, 0], mesh.tri_dir[:, 1], mesh.tri_dir[:, 2]
        pa, pb, pc_ = P[a], P[b], P[c]
        cross = np.cross(pb - pa, pc_ - pa)
        A_tri = 0.5 * np.sqrt(np.sum(cross**2, axis=1))
        fq = bc.q_flux * A_tri / 3.0
        np.add.at(f, a, fq); np.add.at(f, b, fq); np.add.at(f, c, fq)
        dir_nodes = np.array([], dtype=int)
        free_idx = np.arange(Nn)
        Nf = Nn
        Nd = 0
        if log:
            log(f"Neumann BC: q_flux={bc.q_flux} W/m², all nodes free (Nf={Nf})")
        K_ff = K_full
        M_ff = M_full
        f_free_static = f.copy()
    else:
        # Dirichlet partition
        dir_nodes = np.unique(mesh.tri_dir.ravel())
        free_mask = np.ones(Nn, dtype=bool)
        free_mask[dir_nodes] = False
        free_idx = np.where(free_mask)[0]
        Nf, Nd = len(free_idx), len(dir_nodes)

        if log:
            log(f"Dirichlet: {Nd}, Free: {Nf}")

        K_ff = K_full[free_idx, :][:, free_idx]
        M_ff = M_full[free_idx, :][:, free_idx]
        K_fd_g = K_full[free_idx, :][:, dir_nodes] @ np.full(Nd, bc.T_wall)
        f_free_static = f[free_idx] - K_fd_g

    dt = sp.dt
    A_ff = (M_ff / dt) + K_ff

    if log:
        log("Factorizing system matrix ...")
    t0 = timer.time()
    solve_A = factorized(A_ff.tocsc())
    if log:
        log(f"Factorized in {timer.time()-t0:.1f}s")

    # Time stepping
    Tn_free = np.full(Nf, bc.T0, dtype=float)
    n_max = int(round(sp.t_end / dt))

    # Animation capture schedule (coarser than 2D due to larger mesh)
    t_dense_end = min(2.0, sp.t_end)
    dt_dense  = max(dt * 2, 0.02)
    dt_coarse = max(dt * 10, 0.5)
    _anim_targets = np.concatenate([
        np.arange(0.0, t_dense_end + 1e-12, dt_dense),
        np.arange(t_dense_end + dt_coarse, sp.t_end + 1e-12, dt_coarse)
    ])
    anim_set = set(int(round(ta / dt)) for ta in _anim_targets)
    anim_set.add(0)

    T_history = []
    times_anim = []
    tmax_list = []
    tavg_list = []
    tmin_list = []
    times_list = []
    dT_hist_list = []
    converged = False
    conv_step = -1
    last_dT = float('inf')

    if log:
        log(f"Time stepping: up to {n_max} steps ...")

    t_start = timer.time()
    for n in range(n_max + 1):
        t = n * dt
        if bc.bc_inner == "neumann":
            Tn_full_snap = Tn_free.copy()
        else:
            Tn_full_snap = np.full(Nn, bc.T_wall, dtype=float)
            Tn_full_snap[free_idx] = Tn_free
        tmax_list.append(Tn_full_snap.max())
        tavg_list.append(Tn_full_snap.mean())
        tmin_list.append(Tn_full_snap.min())
        times_list.append(t)

        if n in anim_set:
            T_history.append(Tn_free.copy())
            times_anim.append(t)

        if progress and n % 10 == 0:
            progress(n, n_max, last_dT, tmax_list[-1])

        if n == n_max:
            break

        rhs = f_free_static + (M_ff / dt) @ Tn_free
        Tn_free_new = solve_A(rhs)
        dT_max = np.max(np.abs(Tn_free_new - Tn_free))
        last_dT = float(dT_max)
        dT_hist_list.append(last_dT)
        Tn_free = Tn_free_new

        if dT_max < sp.conv_tol and n > 0:
            conv_step = n + 1
            t_conv = (n + 1) * dt
            if bc.bc_inner == "neumann":
                Tn_full_snap = Tn_free.copy()
            else:
                Tn_full_snap = np.full(Nn, bc.T_wall, dtype=float)
                Tn_full_snap[free_idx] = Tn_free
            tmax_list.append(Tn_full_snap.max())
            tavg_list.append(Tn_full_snap.mean())
            tmin_list.append(Tn_full_snap.min())
            times_list.append(t_conv)
            T_history.append(Tn_free.copy())
            times_anim.append(t_conv)
            converged = True
            elapsed = timer.time() - t_start
            if log:
                log(f"CONVERGED at step {conv_step}, t={t_conv:.3f}s "
                    f"[{elapsed:.1f}s] (max|dT|={dT_max:.2e})")
            break

    tmax_hist = np.array(tmax_list)
    tavg_hist = np.array(tavg_list)
    tmin_hist = np.array(tmin_list)
    times_arr = np.array(times_list)

    if bc.bc_inner == "neumann":
        Tn_full = Tn_free.copy()
    else:
        Tn_full = np.full(Nn, bc.T_wall, dtype=float)
        Tn_full[free_idx] = Tn_free

    info = (f"T_min={Tn_full.min():.4f}C, T_max={Tn_full.max():.4f}C\n"
            f"{'Converged' if converged else 'Not converged'} "
            f"at t={times_arr[-1]:.3f}s")
    if log:
        log(info)

    # Mirror
    P_full, T_full, tets_full = mirror_3d(P, Tn_full, mesh.all_tets, geom.xc)

    # make_full_T helper for GIF generation (half-domain T_free → full Tn_full)
    if bc.bc_inner == "neumann":
        def make_full_T(T_free_frame):
            return T_free_frame.copy()
    else:
        def make_full_T(T_free_frame):
            Th = np.full(Nn, bc.T_wall, dtype=float)
            Th[free_idx] = T_free_frame
            return Th

    result = SolveResult(
        T=Tn_full, T_full=T_full, coords_full=P_full, tets_full=tets_full,
        T_min=Tn_full.min(), T_max=Tn_full.max(), q_in=0.0,
        tmax_hist=tmax_hist, tavg_hist=tavg_hist, tmin_hist=tmin_hist,
        times_arr=times_arr,
        converged=converged, conv_step=conv_step,
        T_history=T_history, times_anim=times_anim,
        dT_hist=np.array(dT_hist_list) if dT_hist_list else None,
        info=info
    )
    result._make_full_T = make_full_T
    result._free_idx = free_idx
    return result
