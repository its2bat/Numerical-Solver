"""2D Transient FEM solver with convergence auto-stop."""
import numpy as np
from scipy.sparse.linalg import factorized
from .params import *
from .fem_core import (assemble_KM_2d, apply_robin_2d,
                       build_sparse, mirror_2d)


def solve_transient_2d(mesh: MeshResult, mat: MaterialParams,
                       bc: BCParams, geom: GeometryParams,
                       sp: SolverParams,
                       log=None, progress=None) -> SolveResult:
    """
    Backward Euler transient solver for 2D.
    progress(step, n_max, dT_max, Tmax) is called each step.
    """
    coords = mesh.coords
    Nn = mesh.n_nodes

    f_full = np.zeros(Nn)
    K_rows, K_cols, K_vals = [], [], []
    M_rows, M_cols, M_vals = [], [], []

    for tri, k, rho, cp, name in [
        (mesh.tris_tube, mat.k_tube, mat.rho_tube, mat.cp_tube, "tube"),
        (mesh.tris_paste, mat.k_paste, mat.rho_paste, mat.cp_paste, "paste"),
        (mesh.tris_mold, mat.k_solder, mat.rho_solder, mat.cp_solder, "mold"),
    ]:
        kr, kc, kv, mr, mc, mv = assemble_KM_2d(tri, coords, k, rho, cp)
        K_rows.append(kr); K_cols.append(kc); K_vals.append(kv)
        M_rows.append(mr); M_cols.append(mc); M_vals.append(mv)
        if log:
            log(f"  assembled {name}: {len(tri)} elements")

    # Robin BC
    rob_r, rob_c, rob_v, f_rob = apply_robin_2d(
        mesh.edges_rob, coords, bc.h_conv, bc.Tinf)
    K_rows.append(rob_r); K_cols.append(rob_c); K_vals.append(rob_v)
    f_full += f_rob

    K_full = build_sparse(K_rows, K_cols, K_vals, Nn)
    M_full = build_sparse(M_rows, M_cols, M_vals, Nn)

    # Neumann flux on inner arc (if requested)
    if bc.bc_inner == "neumann":
        ei, ej = mesh.edges_dir[:, 0], mesh.edges_dir[:, 1]
        dx = coords[ej, 0] - coords[ei, 0]
        dy = coords[ej, 1] - coords[ei, 1]
        Le = np.hypot(dx, dy)
        np.add.at(f_full, ei, bc.q_flux * Le / 2.0)
        np.add.at(f_full, ej, bc.q_flux * Le / 2.0)
        arc_nodes = np.array([], dtype=int)
        free_idx = np.arange(Nn)
        Nf = Nn
        Nd = 0
        if log:
            log(f"Neumann BC: q_flux={bc.q_flux} W/m², all nodes free (Nf={Nf})")
        K_ff = K_full
        M_ff = M_full
        f_free_static = f_full.copy()
    else:
        # Dirichlet partition
        arc_nodes = np.unique(mesh.edges_dir.ravel())
        free_mask = np.ones(Nn, dtype=bool)
        free_mask[arc_nodes] = False
        free_idx = np.where(free_mask)[0]
        Nf, Nd = len(free_idx), len(arc_nodes)

        if log:
            log(f"Dirichlet: {Nd}, Free: {Nf}")

        K_ff = K_full[free_idx, :][:, free_idx]
        M_ff = M_full[free_idx, :][:, free_idx]

        K_fd_g = K_full[free_idx, :][:, arc_nodes] @ np.full(Nd, bc.T_wall)
        f_free_static = f_full[free_idx] - K_fd_g

    dt = sp.dt
    A_ff = (M_ff / dt) + K_ff

    if log:
        log("Factorizing system matrix ...")
    solve_A = factorized(A_ff.tocsc())
    if log:
        log("Factorized.")

    # Time stepping
    Tn_free = np.full(Nf, bc.T0, dtype=float)
    n_max = int(round(sp.t_end / dt))

    # Animation capture schedule
    t_dense_end = 2.0
    dt_dense = 0.01
    dt_coarse = 0.5
    times_anim_target = np.concatenate([
        np.arange(0.0, t_dense_end + 1e-12, dt_dense),
        np.arange(2.5, sp.t_end + 1e-12, dt_coarse)
    ])
    anim_set = set(int(round(t / dt)) for t in times_anim_target)
    anim_set.add(0)

    T_history = []
    times_anim = []
    tmax_list = []
    times_list = []
    dT_hist_list = []
    converged = False
    conv_step = -1
    last_dT = float('inf')

    if log:
        log(f"Time stepping: up to {n_max} steps, conv_tol={sp.conv_tol} ...")

    for n in range(n_max + 1):
        t = n * dt
        Tn_full = np.full(Nn, bc.T_wall, dtype=float)
        Tn_full[free_idx] = Tn_free
        tmax_list.append(Tn_full.max())
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
            Tn_full = np.full(Nn, bc.T_wall, dtype=float)
            Tn_full[free_idx] = Tn_free
            tmax_list.append(Tn_full.max())
            times_list.append(t_conv)
            T_history.append(Tn_free.copy())
            times_anim.append(t_conv)
            converged = True
            if log:
                log(f"CONVERGED at step {conv_step}, t={t_conv:.3f}s "
                    f"(max|dT|={dT_max:.2e})")
            break

    tmax_hist = np.array(tmax_list)
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
    coords_full, T_full_arr, tris_full = mirror_2d(
        coords, Tn_full, mesh.triangles, geom.xc)

    # Build make_full helper for animation frames
    non_sym = ~np.isclose(coords[:, 0], geom.xc, atol=1e-12)
    non_sym_idx = np.where(non_sym)[0]
    mirror_map = np.arange(Nn, dtype=int)
    n_mirror = np.sum(non_sym)
    mirror_map[non_sym] = Nn + np.arange(n_mirror)

    if bc.bc_inner == "neumann":
        def make_full_T(T_free_frame):
            Thalf = T_free_frame.copy()
            Tfull = np.empty(coords_full.shape[0])
            Tfull[:Nn] = Thalf
            Tfull[mirror_map[non_sym_idx]] = Thalf[non_sym_idx]
            return Tfull
    else:
        def make_full_T(T_free_frame):
            Thalf = np.full(Nn, bc.T_wall, dtype=float)
            Thalf[free_idx] = T_free_frame
            Tfull = np.empty(coords_full.shape[0])
            Tfull[:Nn] = Thalf
            Tfull[mirror_map[non_sym_idx]] = Thalf[non_sym_idx]
            return Tfull

    # Store make_full function and frame data for animation
    result = SolveResult(
        T=Tn_full, T_full=T_full_arr, coords_full=coords_full,
        tris_full=tris_full,
        T_min=Tn_full.min(), T_max=Tn_full.max(),
        tmax_hist=tmax_hist, times_arr=times_arr,
        converged=converged, conv_step=conv_step,
        T_history=T_history, times_anim=times_anim,
        dT_hist=np.array(dT_hist_list) if dT_hist_list else None,
        info=info
    )
    result._make_full_T = make_full_T
    result._free_idx = free_idx
    return result
