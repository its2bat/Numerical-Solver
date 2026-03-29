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

    tmax_list = []
    times_list = []
    converged = False
    conv_step = -1

    if log:
        log(f"Time stepping: up to {n_max} steps ...")

    t_start = timer.time()
    for n in range(n_max + 1):
        t = n * dt
        Tn_full = np.full(Nn, bc.T_wall, dtype=float)
        Tn_full[free_idx] = Tn_free
        tmax_list.append(Tn_full.max())
        times_list.append(t)

        if progress and n % 10 == 0:
            progress(n, n_max, 0.0, tmax_list[-1])

        if n == n_max:
            break

        rhs = f_free_static + (M_ff / dt) @ Tn_free
        Tn_free_new = solve_A(rhs)
        dT_max = np.max(np.abs(Tn_free_new - Tn_free))
        Tn_free = Tn_free_new

        if dT_max < sp.conv_tol and n > 0:
            conv_step = n + 1
            t_conv = (n + 1) * dt
            Tn_full = np.full(Nn, bc.T_wall, dtype=float)
            Tn_full[free_idx] = Tn_free
            tmax_list.append(Tn_full.max())
            times_list.append(t_conv)
            converged = True
            elapsed = timer.time() - t_start
            if log:
                log(f"CONVERGED at step {conv_step}, t={t_conv:.3f}s "
                    f"[{elapsed:.1f}s] (max|dT|={dT_max:.2e})")
            break

    tmax_hist = np.array(tmax_list)
    times_arr = np.array(times_list)

    Tn_full = np.full(Nn, bc.T_wall, dtype=float)
    Tn_full[free_idx] = Tn_free

    info = (f"T_min={Tn_full.min():.4f}C, T_max={Tn_full.max():.4f}C\n"
            f"{'Converged' if converged else 'Not converged'} "
            f"at t={times_arr[-1]:.3f}s")
    if log:
        log(info)

    # Mirror
    P_full, T_full, tets_full = mirror_3d(P, Tn_full, mesh.all_tets, geom.xc)

    return SolveResult(
        T=Tn_full, T_full=T_full, coords_full=P_full, tets_full=tets_full,
        T_min=Tn_full.min(), T_max=Tn_full.max(), q_in=0.0,
        tmax_hist=tmax_hist, times_arr=times_arr,
        converged=converged, conv_step=conv_step,
        info=info
    )
