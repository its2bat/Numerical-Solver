"""
Plane-stress linear elasticity FEM solver.

DOF ordering
------------
Node i → global DOFs [2i, 2i+1] = [u_x, u_y].

Constitutive matrix (plane stress)
-----------------------------------
         E       ┌ 1   ν      0      ┐
C = ─────────── │ ν   1      0      │
     1 − ν²     └ 0   0  (1−ν)/2   ┘

Element stiffness (P1 triangle, area A, thickness t)
------------------------------------------------------
Ke = t · A · Bᵀ C B      (6×6)

where B (3×6) is the constant strain–displacement matrix.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# ── stiffness assembly ────────────────────────────────────────────────────────

def assemble_plane_stress(tris, coords, E, nu, t, log=None):
    """Vectorised plane-stress stiffness matrix.

    Parameters
    ----------
    tris   : (Ne, 3) int  — triangle connectivity
    coords : (Nn, 2) float — node coordinates (x, y)
    E, nu  : Young's modulus (Pa), Poisson ratio (–)
    t      : out-of-plane thickness (m)

    Returns
    -------
    K : (2Nn, 2Nn) CSR sparse stiffness matrix
    """
    Nn = coords.shape[0]

    # Constitutive matrix
    fac = E / (1.0 - nu**2)
    C = fac * np.array([
        [1.0,  nu,          0.0],
        [nu,   1.0,         0.0],
        [0.0,  0.0,  (1.0 - nu) / 2.0],
    ])

    n0, n1, n2 = tris[:, 0], tris[:, 1], tris[:, 2]
    x0, y0 = coords[n0, 0], coords[n0, 1]
    x1, y1 = coords[n1, 0], coords[n1, 1]
    x2, y2 = coords[n2, 0], coords[n2, 1]

    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    A   = 0.5 * np.abs(det)
    valid = A > 1e-30
    if not np.any(valid):
        raise RuntimeError("All elements are degenerate (zero area).")

    A  = A[valid];  det = det[valid]
    n0 = n0[valid]; n1 = n1[valid]; n2 = n2[valid]
    x0 = x0[valid]; y0 = y0[valid]
    x1 = x1[valid]; y1 = y1[valid]
    x2 = x2[valid]; y2 = y2[valid]

    two_A = 2.0 * A

    # Shape-function spatial gradients (Ne,)
    # For CCW orientation (det > 0) these are correct; det sign cancels in Bᵀ C B.
    b0 = (y1 - y2) / two_A;  b1 = (y2 - y0) / two_A;  b2 = (y0 - y1) / two_A
    c0 = (x2 - x1) / two_A;  c1 = (x0 - x2) / two_A;  c2 = (x1 - x0) / two_A

    # B matrix shape (Ne, 3, 6)
    Ne_v = A.shape[0]
    B = np.zeros((Ne_v, 3, 6))
    B[:, 0, 0] = b0;  B[:, 0, 2] = b1;  B[:, 0, 4] = b2
    B[:, 1, 1] = c0;  B[:, 1, 3] = c1;  B[:, 1, 5] = c2
    B[:, 2, 0] = c0;  B[:, 2, 1] = b0
    B[:, 2, 2] = c1;  B[:, 2, 3] = b1
    B[:, 2, 4] = c2;  B[:, 2, 5] = b2

    # Ke = t · A · Bᵀ C B   →  (Ne, 6, 6)
    CB  = np.einsum('ij,ejk->eik', C, B)           # (Ne,3,6)
    BT  = np.transpose(B, (0, 2, 1))               # (Ne,6,3)
    Ke  = (t * A)[:, None, None] * np.einsum('eij,ejk->eik', BT, CB)

    # DOF indices (Ne, 6)
    dofs = np.column_stack([2*n0, 2*n0+1, 2*n1, 2*n1+1, 2*n2, 2*n2+1])

    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs,   (1, 6)).ravel()
    vals = Ke.ravel()

    K = coo_matrix((vals, (rows.astype(int), cols.astype(int))),
                   shape=(2 * Nn, 2 * Nn)).tocsr()

    if log:
        log(f"  Stiffness matrix: {K.shape[0]}×{K.shape[1]}, "
            f"nnz = {K.nnz}")
    return K


# ── load vector ───────────────────────────────────────────────────────────────

def tip_load_vector(Nn, tip_nodes, F_total, direction=1):
    """Distribute F_total uniformly over tip_nodes.

    Parameters
    ----------
    Nn        : total number of nodes
    tip_nodes : 1-D array of node indices
    F_total   : total applied force (N), positive = + direction
    direction : 0 → x,  1 → y  (default: transverse y)

    Returns
    -------
    f : (2*Nn,) load vector
    """
    f = np.zeros(2 * Nn)
    f_per = F_total / len(tip_nodes)
    for nd in tip_nodes:
        f[2 * nd + direction] += f_per
    return f


# ── solver ────────────────────────────────────────────────────────────────────

def solve_beam(K, f, fixed_nodes, log=None):
    """Solve Ku = f with zero-displacement Dirichlet BCs at fixed_nodes.

    Uses the elimination (partition) method:
      K_ff · u_f = f_f   (only free DOFs)

    Returns
    -------
    u : (2*Nn,) displacement vector [u_x0, u_y0, u_x1, u_y1, ...]
    """
    Nn_dof   = K.shape[0]
    fixed_x  = 2 * fixed_nodes
    fixed_y  = 2 * fixed_nodes + 1
    fixed_dofs = np.concatenate([fixed_x, fixed_y])
    free_dofs  = np.setdiff1d(np.arange(Nn_dof), fixed_dofs)

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    f_f  = f[free_dofs]

    if log:
        log(f"  Solving {len(free_dofs)} free DOFs "
            f"({len(fixed_dofs)} constrained)")

    u = np.zeros(Nn_dof)
    u[free_dofs] = spsolve(K_ff, f_f)
    return u


# ── post-processing helpers ───────────────────────────────────────────────────

def tip_displacement(u, tip_nodes, direction=1):
    """Mean displacement at tip nodes in given direction."""
    return float(np.mean(u[2 * tip_nodes + direction]))


def spring_constant(F_total, delta_tip):
    """k = F / δ  (N/m)."""
    if abs(delta_tip) < 1e-30:
        return np.inf
    return F_total / delta_tip


def von_mises(u, tris, coords, E, nu):
    """Element-wise von-Mises stress (Pa).

    Returns array of shape (Ne,) aligned with tris.
    """
    n0, n1, n2 = tris[:, 0], tris[:, 1], tris[:, 2]
    x0, y0 = coords[n0, 0], coords[n0, 1]
    x1, y1 = coords[n1, 0], coords[n1, 1]
    x2, y2 = coords[n2, 0], coords[n2, 1]

    det   = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0)
    A     = 0.5 * np.abs(det)
    valid = A > 1e-30
    Ne    = tris.shape[0]

    two_A = 2.0 * A[valid]
    b0 = (y1[valid] - y2[valid]) / two_A
    b1 = (y2[valid] - y0[valid]) / two_A
    b2 = (y0[valid] - y1[valid]) / two_A
    c0 = (x2[valid] - x1[valid]) / two_A
    c1 = (x0[valid] - x2[valid]) / two_A
    c2 = (x1[valid] - x0[valid]) / two_A

    fac = E / (1.0 - nu**2)
    C = fac * np.array([
        [1.0,  nu,          0.0],
        [nu,   1.0,         0.0],
        [0.0,  0.0, (1.0 - nu) / 2.0],
    ])

    n0v = n0[valid]; n1v = n1[valid]; n2v = n2[valid]
    u_e = np.column_stack([
        u[2*n0v], u[2*n0v+1],
        u[2*n1v], u[2*n1v+1],
        u[2*n2v], u[2*n2v+1],
    ])   # (Nv, 6)

    # ε = B · u_e  → (Nv, 3)
    eps = np.column_stack([
        b0 * u_e[:, 0] + b1 * u_e[:, 2] + b2 * u_e[:, 4],
        c0 * u_e[:, 1] + c1 * u_e[:, 3] + c2 * u_e[:, 5],
        c0 * u_e[:, 0] + b0 * u_e[:, 1] +
        c1 * u_e[:, 2] + b1 * u_e[:, 3] +
        c2 * u_e[:, 4] + b2 * u_e[:, 5],
    ])

    # σ = C ε  → (Nv, 3)
    sigma = eps @ C.T

    sx, sy, txy = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    svm_valid = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)

    svm = np.zeros(Ne)
    svm[valid] = svm_valid
    return svm
