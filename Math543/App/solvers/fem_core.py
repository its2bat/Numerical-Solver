"""Core FEM assembly and solver routines shared by 2D and 3D solvers."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, factorized


# ---- 2D P1 triangle assembly ----

def assemble_stiffness_2d(tris, coords, k_val):
    """Vectorized stiffness for P1 triangles with conductivity k_val."""
    if len(tris) == 0:
        return np.array([]), np.array([]), np.array([])
    i1, i2, i3 = tris[:, 0], tris[:, 1], tris[:, 2]
    x1, y1 = coords[i1, 0], coords[i1, 1]
    x2, y2 = coords[i2, 0], coords[i2, 1]
    x3, y3 = coords[i3, 0], coords[i3, 1]

    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * np.abs(det)
    v = area > 0

    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)[v]
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)[v]
    coeff = k_val / (4.0 * area[v])

    Ke = coeff[:, None, None] * (
        b[:, :, None] * b[:, None, :] +
        c[:, :, None] * c[:, None, :]
    )

    nd = tris[v]
    rows = np.repeat(nd, 3, axis=1).ravel()
    cols = np.tile(nd, (1, 3)).ravel()
    return rows, cols, Ke.ravel()


_M2_TEMPLATE = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)


def assemble_KM_2d(tris, coords, k_val, rho_val, cp_val):
    """Vectorized stiffness + consistent mass for P1 triangles."""
    if len(tris) == 0:
        return (np.array([]),) * 6
    i1, i2, i3 = tris[:, 0], tris[:, 1], tris[:, 2]
    x1, y1 = coords[i1, 0], coords[i1, 1]
    x2, y2 = coords[i2, 0], coords[i2, 1]
    x3, y3 = coords[i3, 0], coords[i3, 1]

    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * np.abs(det)
    v = area > 0

    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)[v]
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)[v]

    coeff_K = k_val / (4.0 * area[v])
    Ke = coeff_K[:, None, None] * (
        b[:, :, None] * b[:, None, :] +
        c[:, :, None] * c[:, None, :]
    )

    coeff_M = (rho_val * cp_val) * area[v] / 12.0
    Me = coeff_M[:, None, None] * _M2_TEMPLATE[None, :, :]

    nd = tris[v]
    rows = np.repeat(nd, 3, axis=1).ravel()
    cols = np.tile(nd, (1, 3)).ravel()
    return rows, cols, Ke.ravel(), rows, cols, Me.ravel()


def apply_robin_2d(edges, coords, h_conv, Tinf):
    """Robin BC for 2D edges. Returns K_rows, K_cols, K_vals, f_add."""
    Nn = coords.shape[0]
    f_add = np.zeros(Nn)
    ei, ej = edges[:, 0], edges[:, 1]
    dx = coords[ej, 0] - coords[ei, 0]
    dy = coords[ej, 1] - coords[ei, 1]
    Le = np.hypot(dx, dy)

    fh = h_conv * Tinf * Le / 2.0
    np.add.at(f_add, ei, fh)
    np.add.at(f_add, ej, fh)

    h_diag = h_conv * Le * (2.0 / 6.0)
    h_off = h_conv * Le * (1.0 / 6.0)
    rows = np.concatenate([ei, ei, ej, ej])
    cols = np.concatenate([ei, ej, ei, ej])
    vals = np.concatenate([h_diag, h_off, h_off, h_diag])
    return rows, cols, vals, f_add


# ---- 3D linear tetrahedron assembly ----

_G_REF = np.array([[-1., -1., -1.], [1., 0., 0.],
                    [0., 1., 0.], [0., 0., 1.]])  # (4, 3)


def assemble_K_tet(tets, P, k_val):
    """Vectorized stiffness for linear tetrahedra."""
    if len(tets) == 0:
        return np.array([]), np.array([]), np.array([])

    i1, i2, i3, i4 = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]
    d1 = P[i2] - P[i1]
    d2 = P[i3] - P[i1]
    d3 = P[i4] - P[i1]
    J = np.stack([d1, d2, d3], axis=2)

    detJ = np.linalg.det(J)
    V = np.abs(detJ) / 6.0
    valid = V > 0

    J_inv = np.linalg.inv(J[valid])
    V_v = V[valid]
    tets_v = tets[valid]

    G_phys = np.einsum('ij,ejk->eik', _G_REF, J_inv)
    Ke = k_val * V_v[:, None, None] * np.einsum('eij,ekj->eik', G_phys, G_phys)

    nd = tets_v
    rows = np.repeat(nd, 4, axis=1).ravel()
    cols = np.tile(nd, (1, 4)).ravel()
    return rows, cols, Ke.ravel()


def assemble_KM_tet(tets, P, k_val, rho_val, cp_val):
    """Vectorized stiffness + consistent mass for linear tetrahedra."""
    if len(tets) == 0:
        return (np.array([]),) * 6

    i1, i2, i3, i4 = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]
    d1, d2, d3 = P[i2] - P[i1], P[i3] - P[i1], P[i4] - P[i1]
    J = np.stack([d1, d2, d3], axis=2)

    detJ = np.linalg.det(J)
    V = np.abs(detJ) / 6.0
    valid = V > 0

    J_inv = np.linalg.inv(J[valid])
    V_v = V[valid]
    tets_v = tets[valid]

    G_phys = np.einsum('ij,ejk->eik', _G_REF, J_inv)
    Ke = k_val * V_v[:, None, None] * np.einsum('eij,ekj->eik', G_phys, G_phys)

    # Mass: rho*cp * V/20 * (1+delta_ij)
    M_t = np.ones((4, 4), dtype=float)
    np.fill_diagonal(M_t, 2.0)
    coeff_M = (rho_val * cp_val) * V_v / 20.0
    Me = coeff_M[:, None, None] * M_t[None, :, :]

    nd = tets_v
    rows = np.repeat(nd, 4, axis=1).ravel()
    cols = np.tile(nd, (1, 4)).ravel()
    return rows, cols, Ke.ravel(), rows, cols, Me.ravel()


_H3_TEMPLATE = np.array([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]])


def apply_robin_3d(tri_rob, P, h_conv, Tinf):
    """Robin BC for 3D boundary triangles. Returns K_rows, K_cols, K_vals, f_add."""
    Nn = P.shape[0]
    f_add = np.zeros(Nn)
    if len(tri_rob) == 0:
        return np.array([]), np.array([]), np.array([]), f_add

    a, b, c = tri_rob[:, 0], tri_rob[:, 1], tri_rob[:, 2]
    pa, pb, pc_ = P[a], P[b], P[c]
    cross = np.cross(pb - pa, pc_ - pa)
    A_tri = 0.5 * np.sqrt(np.sum(cross**2, axis=1))

    coeff = h_conv * A_tri / 12.0
    Kh = coeff[:, None, None] * _H3_TEMPLATE[None, :, :]

    nd = tri_rob
    rows = np.repeat(nd, 3, axis=1).ravel()
    cols = np.tile(nd, (1, 3)).ravel()

    fh = h_conv * Tinf * A_tri / 3.0
    np.add.at(f_add, a, fh)
    np.add.at(f_add, b, fh)
    np.add.at(f_add, c, fh)

    return rows, cols, Kh.ravel(), f_add


def build_sparse(rows_list, cols_list, vals_list, Nn):
    r = np.concatenate(rows_list)
    c = np.concatenate(cols_list)
    v = np.concatenate(vals_list)
    return coo_matrix((v, (r.astype(int), c.astype(int))), shape=(Nn, Nn)).tocsr()


# ---- Mirror utilities ----

def mirror_2d(coords, T, triangles, xc):
    """Mirror half-domain to full domain (2D)."""
    Nn = coords.shape[0]
    tol = 1e-12
    on_sym = np.isclose(coords[:, 0], xc, atol=tol)
    non_sym = ~on_sym
    n_mirror = np.sum(non_sym)

    coords_full = np.empty((Nn + n_mirror, 2))
    coords_full[:Nn] = coords
    coords_full[Nn:, 0] = 2.0 * xc - coords[non_sym, 0]
    coords_full[Nn:, 1] = coords[non_sym, 1]

    T_full = np.empty(Nn + n_mirror)
    T_full[:Nn] = T
    T_full[Nn:] = T[non_sym]

    mirror_map = np.arange(Nn, dtype=int)
    mirror_map[non_sym] = Nn + np.arange(n_mirror)

    tri_m = np.column_stack([mirror_map[triangles[:, 0]],
                              mirror_map[triangles[:, 2]],
                              mirror_map[triangles[:, 1]]])
    tris_full = np.vstack([triangles, tri_m])

    return coords_full, T_full, tris_full


def mirror_3d(P, T, all_tets, xc):
    """Mirror half-domain to full domain (3D)."""
    Nn = P.shape[0]
    on_sym = np.isclose(P[:, 0], xc, atol=1e-12)
    non_sym = ~on_sym
    n_mirror = np.sum(non_sym)

    P_full = np.empty((Nn + n_mirror, 3))
    P_full[:Nn] = P
    P_full[Nn:, 0] = 2.0 * xc - P[non_sym, 0]
    P_full[Nn:, 1] = P[non_sym, 1]
    P_full[Nn:, 2] = P[non_sym, 2]

    T_full = np.empty(Nn + n_mirror)
    T_full[:Nn] = T
    T_full[Nn:] = T[non_sym]

    mirror_map = np.arange(Nn, dtype=int)
    mirror_map[non_sym] = Nn + np.arange(n_mirror)

    tets_m = np.column_stack([
        mirror_map[all_tets[:, 0]],
        mirror_map[all_tets[:, 2]],
        mirror_map[all_tets[:, 1]],
        mirror_map[all_tets[:, 3]],
    ])
    tets_full = np.vstack([all_tets, tets_m])

    return P_full, T_full, tets_full
