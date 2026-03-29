# ================================================================
# Math 543 – BVP Project
# STEADY-STATE 2D HEAT CONDUCTION (FEM) — DIRICHLET INNER WALL
#
# Physical setup:
#   Microtube: OD=0.8 mm, ID=0.5 mm, stainless steel
#   The solid block sits on top of the tube.
#   Domain models tube wall + solid block together (same material).
#
# Geometry (half-domain by symmetry):
#   Rectangle: xc <= x <= W, 0 <= y <= H
#   Quarter-circle cut at R_inner (tube inner wall)
#
# Boundary conditions:
#   INNER ARC (tube inner wall) : T = T_wall          [Dirichlet]
#   OUTER surfaces (3 edges)    : -k dT/dn = h(T-Tinf) [Robin]
#   SYMMETRY (x = xc)           : dT/dn = 0            [natural]
#
# FEM approach:
#   DOFs split into free (f) and Dirichlet (d) sets.
#   Solve: K_ff * T_free = f_free - K_fd * T_wall
#   Then reconstruct: T[free] = T_free, T[dirichlet] = T_wall
#
# Outputs:
#   Twall_steady_half.png
#   Twall_steady_full.png
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import gmsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# -------------------- PARAMETERS (EDIT HERE) ---------------------
# Geometry (mm -> m)
W_mm = 1.4
H_mm = 1.0

# Tube geometry
tube_OD_mm = 0.8   # outer diameter (mm) — sets where tube contacts solid
tube_ID_mm = 0.5   # inner diameter (mm) — Dirichlet BC applied here
R_inner_mm = tube_ID_mm / 2.0   # 0.25 mm  (arc radius for Dirichlet BC)

# Material: stainless steel (tube wall + solid block, combined domain)
k = 16.0   # W/m-K

# Outer surface convection
h    = 15.0   # W/m^2-K
Tinf = 25.0   # °C

# Prescribed inner wall temperature (the known quantity)
T_wall = 60.0   # °C  <-- set this from experiment / calculation

# Mesh sizing
h_global  = 0.06e-3    # m
h_arc     = 0.008e-3   # m  (fine near inner arc where T is prescribed)
dist_min  = 0.02e-3
dist_max  = 0.15e-3

# Output filenames
half_fig = "Twall_steady_half.png"
full_fig = "Twall_steady_full.png"
# ---------------------------------------------------------------

# Unit conversions
W  = W_mm * 1e-3
H  = H_mm * 1e-3
R  = R_inner_mm * 1e-3   # inner arc radius
xc = W / 2.0              # symmetry plane x = xc

# Geometry validity
if R > (W - xc) + 1e-15:
    raise ValueError(f"R_inner ({R_inner_mm} mm) > W/2 ({W_mm/2} mm). Check dimensions.")
if R > H + 1e-15:
    raise ValueError(f"R_inner ({R_inner_mm} mm) > H ({H_mm} mm). Check dimensions.")

print("=== STEADY 2D — Dirichlet inner wall ===")
print(f"Tube: OD={tube_OD_mm} mm, ID={tube_ID_mm} mm  =>  R_inner={R_inner_mm} mm")
print(f"Domain: W={W_mm} mm, H={H_mm} mm")
print(f"k={k} W/m-K, h={h} W/m²-K, Tinf={Tinf} °C")
print(f"T_wall={T_wall} °C  (Dirichlet BC at inner arc)")

# -------------------- BUILD MESH --------------------
try:
    gmsh.finalize()
except Exception:
    pass

gmsh.initialize()
gmsh.model.add("steady_dirichlet_inner")

p1 = gmsh.model.geo.addPoint(xc + R, 0.0, 0.0, h_arc)
p2 = gmsh.model.geo.addPoint(W,      0.0, 0.0, h_global)
p3 = gmsh.model.geo.addPoint(W,      H,   0.0, h_global)
p4 = gmsh.model.geo.addPoint(xc,     H,   0.0, h_global)
p5 = gmsh.model.geo.addPoint(xc,     R,   0.0, h_arc)
pc = gmsh.model.geo.addPoint(xc,     0.0, 0.0, h_arc)

l_bottom = gmsh.model.geo.addLine(p1, p2)
l_right  = gmsh.model.geo.addLine(p2, p3)
l_top    = gmsh.model.geo.addLine(p3, p4)
l_sym    = gmsh.model.geo.addLine(p4, p5)
c_arc    = gmsh.model.geo.addCircleArc(p5, pc, p1)

cloop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_sym, c_arc])
surf  = gmsh.model.geo.addPlaneSurface([cloop])
gmsh.model.geo.synchronize()

pg_arc   = gmsh.model.addPhysicalGroup(1, [c_arc])
pg_outer = gmsh.model.addPhysicalGroup(1, [l_bottom, l_right, l_top])
pg_sym   = gmsh.model.addPhysicalGroup(1, [l_sym])
pg_dom   = gmsh.model.addPhysicalGroup(2, [surf])

# Refinement near inner arc
fd = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [c_arc])
gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)

ft = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(ft, "InField", fd)
gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_arc)
gmsh.model.mesh.field.setNumber(ft, "SizeMax", h_global)
gmsh.model.mesh.field.setNumber(ft, "DistMin", dist_min)
gmsh.model.mesh.field.setNumber(ft, "DistMax", dist_max)
gmsh.model.mesh.field.setAsBackgroundMesh(ft)

gmsh.model.mesh.generate(2)

# -------------------- EXTRACT MESH --------------------
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
coords_all = node_coords.reshape(-1, 3)[:, :2]
tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
triangles_all = None
for etype, enodes in zip(elem_types, elem_node_tags):
    if etype in (2, 9):
        nn = 3 if etype == 2 else 6
        tri3 = enodes.reshape(-1, nn)[:, :3]
        triangles_all = np.vectorize(tag_to_idx.get)(tri3)
        break

if triangles_all is None:
    gmsh.finalize()
    raise RuntimeError("No triangle elements found.")

def edges_from_pg(phys_id):
    edges = []
    for ent in gmsh.model.getEntitiesForPhysicalGroup(1, phys_id):
        etypes, _, enodes = gmsh.model.mesh.getElements(1, ent)
        for etype, nodes in zip(etypes, enodes):
            if etype == 1:
                edges.append(np.vectorize(tag_to_idx.get)(nodes.reshape(-1, 2)))
            elif etype == 8:
                edges.append(np.vectorize(tag_to_idx.get)(nodes.reshape(-1, 3)[:, :2]))
    if not edges:
        return np.empty((0, 2), dtype=int)
    E = np.vstack(edges)
    return np.unique(np.sort(E, axis=1), axis=0)

edges_arc   = edges_from_pg(pg_arc)
edges_outer = edges_from_pg(pg_outer)
edges_sym   = edges_from_pg(pg_sym)
gmsh.finalize()

# -------------------- ACTIVE NODE REDUCTION --------------------
used = set(triangles_all.ravel()) | set(edges_arc.ravel()) | \
       set(edges_outer.ravel()) | set(edges_sym.ravel())
used = np.array(sorted(used), dtype=int)
new_index = -np.ones(coords_all.shape[0], dtype=int)
new_index[used] = np.arange(len(used))

coords    = coords_all[used]
triangles = new_index[triangles_all]
edges_arc   = new_index[edges_arc]
edges_outer = new_index[edges_outer]

Nn = coords.shape[0]
print(f"Nodes: {Nn},  Triangles: {len(triangles)}")
print(f"Arc edges: {len(edges_arc)},  Outer edges: {len(edges_outer)}")

# -------------------- FEM ASSEMBLY (K, f) — VECTORIZED --------------------
# Note: NO Neumann on arc (Dirichlet is enforced via subspace, not forcing)
f = np.zeros(Nn)

# --- Vectorized element stiffness (P1 triangles) ---
i1 = triangles[:, 0]; i2 = triangles[:, 1]; i3 = triangles[:, 2]
x1, y1 = coords[i1, 0], coords[i1, 1]
x2, y2 = coords[i2, 0], coords[i2, 1]
x3, y3 = coords[i3, 0], coords[i3, 1]

det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
area = 0.5 * np.abs(det)
valid = area > 0
Nt_valid = np.sum(valid)

bvec = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)[valid]  # (Nv, 3)
cvec = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)[valid]  # (Nv, 3)
coeff = k / (4.0 * area[valid])  # (Nv,)

Ke_all = coeff[:, None, None] * (
    bvec[:, :, None] * bvec[:, None, :] +
    cvec[:, :, None] * cvec[:, None, :]
)  # (Nv, 3, 3)

nd_valid = triangles[valid]  # (Nv, 3)
rows_K = np.repeat(nd_valid, 3, axis=1).ravel()
cols_K = np.tile(nd_valid, (1, 3)).ravel()
vals_K = Ke_all.ravel()

# --- Robin convection on OUTER edges ---
ei, ej = edges_outer[:, 0], edges_outer[:, 1]
dx = coords[ej, 0] - coords[ei, 0]
dy = coords[ej, 1] - coords[ei, 1]
Le = np.hypot(dx, dy)

# Forcing: h * Tinf * L / 2 per node
fh_vals = h * Tinf * Le / 2.0
np.add.at(f, ei, fh_vals)
np.add.at(f, ej, fh_vals)

# Stiffness contribution: h * L / 6 * [[2,1],[1,2]]
h_diag = h * Le * (2.0 / 6.0)  # diagonal entries
h_off  = h * Le * (1.0 / 6.0)  # off-diagonal entries
rows_R = np.concatenate([ei, ei, ej, ej])
cols_R = np.concatenate([ei, ej, ei, ej])
vals_R = np.concatenate([h_diag, h_off, h_off, h_diag])

# --- Combine and build sparse K ---
all_rows = np.concatenate([rows_K, rows_R])
all_cols = np.concatenate([cols_K, cols_R])
all_vals = np.concatenate([vals_K, vals_R])

K_full = coo_matrix((all_vals, (all_rows, all_cols)), shape=(Nn, Nn)).tocsr()

# -------------------- DIRICHLET BC (subspace partition) --------------------
# arc_nodes: nodes where T = T_wall (prescribed)
# free_idx:  all other nodes (to be solved)
arc_nodes = np.unique(edges_arc.ravel())
free_mask = np.ones(Nn, dtype=bool)
free_mask[arc_nodes] = False
free_idx = np.where(free_mask)[0]

Nf = len(free_idx)
Nd = len(arc_nodes)
print(f"Dirichlet (arc) nodes: {Nd},  Free nodes to solve: {Nf}")

# Submatrix: K_ff (free-free), K_fd (free-dirichlet)
K_ff = K_full[free_idx, :][:, free_idx]

# RHS for free nodes: f_free - K_fd * T_wall
K_fd_g = K_full[free_idx, :][:, arc_nodes] @ np.full(Nd, T_wall)
f_free = f[free_idx] - K_fd_g

# -------------------- SOLVE --------------------
T_free = spsolve(K_ff, f_free)

# Reconstruct full temperature field
T = np.full(Nn, T_wall, dtype=float)
T[free_idx] = T_free

print(f"T min = {T.min():.4f} °C,  T max = {T.max():.4f} °C")
print(f"T at inner wall (should be {T_wall} °C): min={T[arc_nodes].min():.4f}, max={T[arc_nodes].max():.4f}")

# -------------------- ENERGY CHECK --------------------
# Heat leaving through outer surfaces (Robin)
Tm_outer = 0.5 * (T[edges_outer[:, 0]] + T[edges_outer[:, 1]])
Le_outer = np.hypot(coords[edges_outer[:, 1], 0] - coords[edges_outer[:, 0], 0],
                    coords[edges_outer[:, 1], 1] - coords[edges_outer[:, 0], 1])
q_out = np.sum(h * (Tm_outer - Tinf) * Le_outer)
print(f"Heat dissipated through outer surfaces (per unit thickness): {q_out:.6e} W/m")

# Reaction flux at Dirichlet nodes (heat entering from inner wall)
# q_in = K_dd * T_wall + K_df * T_free - f_d   (residual at constrained DOFs)
q_reaction = K_full[arc_nodes, :] @ T - f[arc_nodes]
q_in_total = np.sum(q_reaction)
print(f"Heat entering through inner wall (reaction, per thickness): {q_in_total:.6e} W/m")
if abs(q_in_total) > 1e-14:
    print(f"Energy balance error: {abs(q_out - q_in_total) / abs(q_in_total):.2%}")

# -------------------- PLOT — HALF DOMAIN --------------------
triang_half = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

fig, ax = plt.subplots(figsize=(6.0, 5.2))
cf = ax.tricontourf(triang_half, T, levels=40, cmap="turbo")
cbar = fig.colorbar(cf, ax=ax, pad=0.03)
cbar.set_label("Temperature (°C)")
ax.triplot(triang_half, linewidth=0.10, alpha=0.15)
ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Steady T — half domain  (T_wall={T_wall}°C, Tinf={Tinf}°C)")
ax.grid(True, alpha=0.2)
padx, pady = 0.00015, 0.00010
ax.set_xlim(xc - 0.00005, W + padx)
ax.set_ylim(-pady, H + pady)
plt.savefig(half_fig, dpi=300, bbox_inches="tight")
plt.show()

# -------------------- MIRROR → FULL DOMAIN (vectorized) --------------------
tol = 1e-12
on_sym = np.isclose(coords[:, 0], xc, atol=tol)
N_half = coords.shape[0]
non_sym = ~on_sym
n_mirror = np.sum(non_sym)
non_sym_idx = np.where(non_sym)[0]

# Pre-allocate mirrored coords and T
coords_full = np.empty((N_half + n_mirror, 2))
coords_full[:N_half] = coords
coords_full[N_half:, 0] = 2.0 * xc - coords[non_sym, 0]
coords_full[N_half:, 1] = coords[non_sym, 1]

T_full = np.empty(N_half + n_mirror)
T_full[:N_half] = T
T_full[N_half:] = T[non_sym]

# Build mirror index map
mirror_index = np.arange(N_half, dtype=int)  # sym nodes map to themselves
mirror_index[non_sym] = N_half + np.arange(n_mirror)

tri_m = np.column_stack([mirror_index[triangles[:, 0]],
                          mirror_index[triangles[:, 2]],
                          mirror_index[triangles[:, 1]]])
triangles_full = np.vstack([triangles, tri_m])
triang_full = mtri.Triangulation(coords_full[:, 0], coords_full[:, 1], triangles_full)

fig, ax = plt.subplots(figsize=(6.6, 5.2))
cf = ax.tricontourf(triang_full, T_full, levels=40, cmap="turbo")
cbar = fig.colorbar(cf, ax=ax, pad=0.03)
cbar.set_label("Temperature (°C)")
ax.triplot(triang_full, linewidth=0.10, alpha=0.15)
ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Steady T — full domain  (T_wall={T_wall}°C, Tinf={Tinf}°C)")
ax.grid(True, alpha=0.2)
ax.set_xlim(-padx, W + padx)
ax.set_ylim(-pady, H + pady)
plt.savefig(full_fig, dpi=300, bbox_inches="tight")
plt.show()
