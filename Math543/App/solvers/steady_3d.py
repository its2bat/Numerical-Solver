"""3D Steady-state FEM solver."""
import numpy as np
from scipy.sparse.linalg import spsolve
from .params import *
from .fem_core import (assemble_K_tet, apply_robin_3d,
                       build_sparse, mirror_3d)


def solve_steady_3d(mesh: MeshResult, mat: MaterialParams,
                    bc: BCParams, geom: GeometryParams,
                    log=None) -> SolveResult:
    P = mesh.coords
    Nn = mesh.n_nodes

    f = np.zeros(Nn)
    K_rows, K_cols, K_vals = [], [], []

    for tets, k, name in [(mesh.tets_tube, mat.k_tube, "tube"),
                           (mesh.tets_paste, mat.k_paste, "paste"),
                           (mesh.tets_mold, mat.k_solder, "mold")]:
        r, c, v = assemble_K_tet(tets, P, k)
        K_rows.append(r); K_cols.append(c); K_vals.append(v)
        if log:
            log(f"  assembled {name}: {len(tets)} tets, k={k}")

    # Robin BC
    rob_r, rob_c, rob_v, f_rob = apply_robin_3d(
        mesh.tri_rob, P, bc.h_conv, bc.Tinf)
    K_rows.append(rob_r); K_cols.append(rob_c); K_vals.append(rob_v)
    f += f_rob

    if bc.bc_inner == "neumann":
        # Apply Neumann flux on inner cylinder faces (tri_dir)
        a, b, c = mesh.tri_dir[:, 0], mesh.tri_dir[:, 1], mesh.tri_dir[:, 2]
        pa, pb, pc_ = P[a], P[b], P[c]
        cross = np.cross(pb - pa, pc_ - pa)
        A_tri = 0.5 * np.sqrt(np.sum(cross**2, axis=1))
        fq = bc.q_flux * A_tri / 3.0
        np.add.at(f, a, fq); np.add.at(f, b, fq); np.add.at(f, c, fq)
        K = build_sparse(K_rows, K_cols, K_vals, Nn)
        if log:
            log(f"Neumann BC: q_flux={bc.q_flux} W/m², full system solve ...")
            log("Solving ...")
        T = spsolve(K, f)
        q_in = float(bc.q_flux) * float(A_tri.sum())
        info = (f"T_min={T.min():.4f}C, T_max={T.max():.4f}C\n"
                f"Heat in: {q_in:.6e} W")
        if log:
            log(info)
    else:
        K = build_sparse(K_rows, K_cols, K_vals, Nn)

        # Dirichlet
        dir_nodes = np.unique(mesh.tri_dir.ravel())
        free_mask = np.ones(Nn, dtype=bool)
        free_mask[dir_nodes] = False
        free_idx = np.where(free_mask)[0]
        Nd = len(dir_nodes)

        if log:
            log(f"Dirichlet: {Nd}, Free: {len(free_idx)}")
            log("Solving ...")

        K_ff = K[free_idx, :][:, free_idx]
        rhs = f[free_idx] - K[free_idx, :][:, dir_nodes] @ np.full(Nd, bc.T_wall)

        T = np.full(Nn, bc.T_wall, dtype=float)
        T[free_idx] = spsolve(K_ff, rhs)

        q_in = np.sum(K[dir_nodes, :] @ T - f[dir_nodes])

        info = (f"T_min={T.min():.4f}C, T_max={T.max():.4f}C\n"
                f"Heat in: {q_in:.6e} W")
        if log:
            log(info)

    # Mirror
    P_full, T_full, tets_full = mirror_3d(P, T, mesh.all_tets, geom.xc)

    return SolveResult(
        T=T, T_full=T_full, coords_full=P_full, tets_full=tets_full,
        T_min=T.min(), T_max=T.max(), q_in=q_in, info=info
    )
