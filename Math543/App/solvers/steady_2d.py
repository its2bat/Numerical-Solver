"""2D Steady-state FEM solver."""
import numpy as np
from scipy.sparse.linalg import spsolve
from .params import *
from .fem_core import (assemble_stiffness_2d, apply_robin_2d,
                       build_sparse, mirror_2d)


def solve_steady_2d(mesh: MeshResult, mat: MaterialParams,
                    bc: BCParams, geom: GeometryParams,
                    log=None) -> SolveResult:
    coords = mesh.coords
    Nn = mesh.n_nodes

    f = np.zeros(Nn)
    K_rows, K_cols, K_vals = [], [], []

    for tri, k, name in [(mesh.tris_tube, mat.k_tube, "tube"),
                          (mesh.tris_paste, mat.k_paste, "paste"),
                          (mesh.tris_mold, mat.k_solder, "mold")]:
        r, c, v = assemble_stiffness_2d(tri, coords, k)
        K_rows.append(r); K_cols.append(c); K_vals.append(v)
        if log:
            log(f"  assembled {name}: {len(tri)} elements, k={k}")

    # Robin BC
    rob_r, rob_c, rob_v, f_rob = apply_robin_2d(
        mesh.edges_rob, coords, bc.h_conv, bc.Tinf)
    K_rows.append(rob_r); K_cols.append(rob_c); K_vals.append(rob_v)
    f += f_rob

    if bc.bc_inner == "neumann":
        # Apply Neumann flux on inner arc edges
        ei, ej = mesh.edges_dir[:, 0], mesh.edges_dir[:, 1]
        dx = coords[ej, 0] - coords[ei, 0]
        dy = coords[ej, 1] - coords[ei, 1]
        Le = np.hypot(dx, dy)
        np.add.at(f, ei, bc.q_flux * Le / 2.0)
        np.add.at(f, ej, bc.q_flux * Le / 2.0)
        # No Dirichlet — full system solve
        K = build_sparse(K_rows, K_cols, K_vals, Nn)
        if log:
            log(f"Neumann BC: q_flux={bc.q_flux} W/m², full system solve ...")
            log("Solving ...")
        T = spsolve(K, f)
        q_in = float(bc.q_flux) * float(sum(np.hypot(
            coords[mesh.edges_dir[:, 1], 0] - coords[mesh.edges_dir[:, 0], 0],
            coords[mesh.edges_dir[:, 1], 1] - coords[mesh.edges_dir[:, 0], 1]
        )))
        Tm_rob = 0.5 * (T[mesh.edges_rob[:, 0]] + T[mesh.edges_rob[:, 1]])
        Le_rob = np.hypot(
            coords[mesh.edges_rob[:, 1], 0] - coords[mesh.edges_rob[:, 0], 0],
            coords[mesh.edges_rob[:, 1], 1] - coords[mesh.edges_rob[:, 0], 1])
        q_out = np.sum(bc.h_conv * (Tm_rob - bc.Tinf) * Le_rob)
        info = (f"T_min = {T.min():.4f} C,  T_max = {T.max():.4f} C\n"
                f"Heat in: {q_in:.6e} W/m | Heat out: {q_out:.6e} W/m")
        if abs(q_in) > 1e-14:
            info += f"\nEnergy balance error: {abs(q_out - q_in)/abs(q_in):.2%}"
        if log:
            log(info)
    else:
        K = build_sparse(K_rows, K_cols, K_vals, Nn)

        # Dirichlet
        arc_nodes = np.unique(mesh.edges_dir.ravel())
        free_mask = np.ones(Nn, dtype=bool)
        free_mask[arc_nodes] = False
        free_idx = np.where(free_mask)[0]
        Nd = len(arc_nodes)

        if log:
            log(f"Dirichlet: {Nd}, Free: {len(free_idx)}")

        K_ff = K[free_idx, :][:, free_idx]
        K_fd_g = K[free_idx, :][:, arc_nodes] @ np.full(Nd, bc.T_wall)
        rhs = f[free_idx] - K_fd_g

        if log:
            log("Solving ...")
        T_free = spsolve(K_ff, rhs)

        T = np.full(Nn, bc.T_wall, dtype=float)
        T[free_idx] = T_free

        # Energy balance
        q_in = np.sum(K[arc_nodes, :] @ T - f[arc_nodes])
        Tm_rob = 0.5 * (T[mesh.edges_rob[:, 0]] + T[mesh.edges_rob[:, 1]])
        Le_rob = np.hypot(
            coords[mesh.edges_rob[:, 1], 0] - coords[mesh.edges_rob[:, 0], 0],
            coords[mesh.edges_rob[:, 1], 1] - coords[mesh.edges_rob[:, 0], 1])
        q_out = np.sum(bc.h_conv * (Tm_rob - bc.Tinf) * Le_rob)

        info = (f"T_min = {T.min():.4f} C,  T_max = {T.max():.4f} C\n"
                f"Heat in: {q_in:.6e} W/m | Heat out: {q_out:.6e} W/m")
        if abs(q_in) > 1e-14:
            info += f"\nEnergy balance error: {abs(q_out - q_in)/abs(q_in):.2%}"
        if log:
            log(info)

    # Mirror
    coords_full, T_full, tris_full = mirror_2d(coords, T, mesh.triangles, geom.xc)

    return SolveResult(
        T=T, T_full=T_full, coords_full=coords_full, tris_full=tris_full,
        T_min=T.min(), T_max=T.max(), q_in=q_in, q_out=q_out, info=info
    )
