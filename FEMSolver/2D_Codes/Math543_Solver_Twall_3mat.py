# ================================================================
# FEM Heat Transfer – BVP Project
# STEADY-STATE 2D FEM — 3-MATERIAL MODEL — FULL TUBE
#
# Physical setup (half-domain, symmetry at x = W/2):
#
#   UPPER REGION (y > 0):  mold (rectangle)
#     Solder mold (60Sn/40Pb, k=50)
#       └─ Thermal paste (annular, k=9)
#            └─ Tube wall upper quarter (SS316, k=16)
#                 └─ Water channel: T = T_wall  [Dirichlet]
#
#   LOWER REGION (y < 0):  exposed tube
#     Tube wall lower quarter (SS316, k=16)
#       inner:  T = T_wall  [Dirichlet]
#       outer:  air convection  [Robin]
#
# BCs:
#   INNER ARC (full half-circle R_in):  T = T_wall       [Dirichlet]
#   MOLD OUTER (bottom/right/top):      h(T-T∞)          [Robin]
#   PASTE BOTTOM (y=0, R_tube→R_paste): h(T-T∞)          [Robin]
#   EXPOSED TUBE ARC (R_tube, y<0):     h(T-T∞)          [Robin]
#   SYMMETRY (x = xc):                  dT/dn = 0        [natural]
#   INTERFACES:                          continuity       [automatic]
#
# Outputs:
#   Twall_3mat_steady_half.png
#   Twall_3mat_steady_full.png
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import gmsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# ======================== PARAMETERS ========================
# Geometry (mm)
W_mm = 1.4
H_mm = 1.0

tube_OD_mm = 0.8
tube_ID_mm = 0.5
paste_thickness_mm = 0.1

R_in_mm    = tube_ID_mm / 2.0                   # 0.25 mm
R_tube_mm  = tube_OD_mm / 2.0                   # 0.40 mm
R_paste_mm = R_tube_mm + paste_thickness_mm      # 0.50 mm

# Material properties
k_tube   = 16.0    # W/mK  — stainless steel 316
k_paste  = 9.0     # W/mK  — thermal paste
k_solder = 50.0    # W/mK  — 60Sn / 40Pb solder

# Outer convection (air)
h_conv = 15.0      # W/m²K
Tinf   = 25.0      # °C

# Prescribed inner wall temperature
T_wall = 60.0      # °C

# Mesh sizing (m)
h_global = 0.06e-3
h_arc    = 0.008e-3
h_mid    = 0.015e-3
dist_min = 0.02e-3
dist_max = 0.15e-3

# Output
half_fig = "Twall_3mat_steady_half.png"
full_fig = "Twall_3mat_steady_full.png"

# ======================== UNIT CONVERSION ========================
W  = W_mm * 1e-3;  H = H_mm * 1e-3
R_in    = R_in_mm    * 1e-3
R_tube  = R_tube_mm  * 1e-3
R_paste = R_paste_mm * 1e-3
xc = W / 2.0

for name, r in [("R_paste", R_paste_mm), ("R_tube", R_tube_mm), ("R_in", R_in_mm)]:
    if r * 1e-3 > (W - xc) + 1e-15:
        raise ValueError(f"{name} ({r} mm) > W/2 ({W_mm/2} mm)")
    if r * 1e-3 > H + 1e-15:
        raise ValueError(f"{name} ({r} mm) > H ({H_mm} mm)")

print("=" * 60)
print("STEADY 2D — 3-material, full tube with exposed section")
print("=" * 60)
print(f"Tube: ID={tube_ID_mm}mm, OD={tube_OD_mm}mm")
print(f"Paste thickness: {paste_thickness_mm}mm")
print(f"Radii: R_in={R_in_mm}mm -> R_tube={R_tube_mm}mm -> R_paste={R_paste_mm}mm")
print(f"k: tube={k_tube}, paste={k_paste}, solder={k_solder} W/mK")
print(f"BC: T_wall={T_wall}C | h={h_conv} W/m2K, T_inf={Tinf}C")

# ================================================================
# 1) BUILD MESH  (gmsh)
# ================================================================
try:
    gmsh.finalize()
except Exception:
    pass

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("steady_3mat_full_tube")

# ---- Points ----
pc = gmsh.model.geo.addPoint(xc, 0.0, 0.0, h_arc)   # tube center

# Inner arc (R_in): top, right, bottom
p_in_top   = gmsh.model.geo.addPoint(xc,        R_in,  0.0, h_arc)
p_in_right = gmsh.model.geo.addPoint(xc + R_in, 0.0,   0.0, h_arc)
p_in_bot   = gmsh.model.geo.addPoint(xc,       -R_in,  0.0, h_arc)

# Tube arc (R_tube): top, right, bottom
p_tube_top   = gmsh.model.geo.addPoint(xc,          R_tube, 0.0, h_mid)
p_tube_right = gmsh.model.geo.addPoint(xc + R_tube, 0.0,    0.0, h_mid)
p_tube_bot   = gmsh.model.geo.addPoint(xc,         -R_tube, 0.0, h_mid)

# Paste arc (R_paste): top, right  (upper region only)
p_paste_top   = gmsh.model.geo.addPoint(xc,           R_paste, 0.0, h_mid)
p_paste_right = gmsh.model.geo.addPoint(xc + R_paste, 0.0,     0.0, h_mid)

# Mold rectangle corners
p_mold_br = gmsh.model.geo.addPoint(W,  0.0, 0.0, h_global)
p_mold_tr = gmsh.model.geo.addPoint(W,  H,   0.0, h_global)
p_mold_tl = gmsh.model.geo.addPoint(xc, H,   0.0, h_global)

# ---- Arcs (quarter circles) ----
arc_in_u   = gmsh.model.geo.addCircleArc(p_in_top,   pc, p_in_right)    # upper inner
arc_in_l   = gmsh.model.geo.addCircleArc(p_in_right,  pc, p_in_bot)      # lower inner
arc_tube_u = gmsh.model.geo.addCircleArc(p_tube_top,  pc, p_tube_right)  # upper tube
arc_tube_l = gmsh.model.geo.addCircleArc(p_tube_right, pc, p_tube_bot)   # lower tube (exposed)
arc_paste  = gmsh.model.geo.addCircleArc(p_paste_top, pc, p_paste_right) # upper paste

# ---- Lines at y=0 ----
l_y0_tube  = gmsh.model.geo.addLine(p_in_right,    p_tube_right)   # internal
l_y0_paste = gmsh.model.geo.addLine(p_tube_right,  p_paste_right)  # paste bottom (Robin)
l_y0_mold  = gmsh.model.geo.addLine(p_paste_right, p_mold_br)      # mold bottom (Robin)

# ---- Mold rectangle edges ----
l_right = gmsh.model.geo.addLine(p_mold_br, p_mold_tr)
l_top   = gmsh.model.geo.addLine(p_mold_tr, p_mold_tl)

# ---- Symmetry lines (x = xc, going down) ----
l_sym_mold   = gmsh.model.geo.addLine(p_mold_tl,  p_paste_top)
l_sym_paste  = gmsh.model.geo.addLine(p_paste_top, p_tube_top)
l_sym_tube_u = gmsh.model.geo.addLine(p_tube_top,  p_in_top)
l_sym_tube_l = gmsh.model.geo.addLine(p_tube_bot,  p_in_bot)   # lower (going up)

# ================================================================
# SURFACES (4 material regions)
# ================================================================

# 1) Tube wall — upper quarter
#    p_in_top → p_in_right → p_tube_right → p_tube_top → p_in_top
cl_tube_u = gmsh.model.geo.addCurveLoop([
    arc_in_u, l_y0_tube, -arc_tube_u, l_sym_tube_u
])
s_tube_u = gmsh.model.geo.addPlaneSurface([cl_tube_u])

# 2) Tube wall — lower quarter
#    p_tube_right → p_tube_bot → p_in_bot → p_in_right → p_tube_right
cl_tube_l = gmsh.model.geo.addCurveLoop([
    arc_tube_l, l_sym_tube_l, -arc_in_l, l_y0_tube
])
s_tube_l = gmsh.model.geo.addPlaneSurface([cl_tube_l])

# 3) Thermal paste — upper quarter
#    p_tube_top → p_tube_right → p_paste_right → p_paste_top → p_tube_top
cl_paste = gmsh.model.geo.addCurveLoop([
    arc_tube_u, l_y0_paste, -arc_paste, l_sym_paste
])
s_paste = gmsh.model.geo.addPlaneSurface([cl_paste])

# 4) Solder mold (rectangle minus paste circle)
#    p_paste_top → p_paste_right → p_mold_br → p_mold_tr → p_mold_tl → p_paste_top
cl_mold = gmsh.model.geo.addCurveLoop([
    arc_paste, l_y0_mold, l_right, l_top, l_sym_mold
])
s_mold = gmsh.model.geo.addPlaneSurface([cl_mold])

gmsh.model.geo.synchronize()

# ---- Physical groups ----
pg_dirichlet = gmsh.model.addPhysicalGroup(1, [arc_in_u, arc_in_l])
pg_robin     = gmsh.model.addPhysicalGroup(1, [
    arc_tube_l,    # exposed tube outer (y < 0)
    l_y0_paste,    # paste bottom (y = 0)
    l_y0_mold,     # mold bottom (y = 0)
    l_right,       # mold right
    l_top          # mold top
])
pg_sym = gmsh.model.addPhysicalGroup(1, [
    l_sym_mold, l_sym_paste, l_sym_tube_u, l_sym_tube_l
])

pg_s_tube_u = gmsh.model.addPhysicalGroup(2, [s_tube_u])
pg_s_tube_l = gmsh.model.addPhysicalGroup(2, [s_tube_l])
pg_s_paste  = gmsh.model.addPhysicalGroup(2, [s_paste])
pg_s_mold   = gmsh.model.addPhysicalGroup(2, [s_mold])

# ---- Mesh refinement near arcs ----
fd = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [
    arc_in_u, arc_in_l, arc_tube_u, arc_tube_l, arc_paste
])
gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)

ft = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(ft, "InField", fd)
gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_arc)
gmsh.model.mesh.field.setNumber(ft, "SizeMax", h_global)
gmsh.model.mesh.field.setNumber(ft, "DistMin", dist_min)
gmsh.model.mesh.field.setNumber(ft, "DistMax", dist_max)
gmsh.model.mesh.field.setAsBackgroundMesh(ft)

gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Laplace2D")

# ================================================================
# 2) EXTRACT MESH
# ================================================================
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
coords_all = node_coords.reshape(-1, 3)[:, :2]
tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}


def get_tris(surf_tag):
    """Triangle connectivity (0-based) from a surface."""
    etypes, _, enodes = gmsh.model.mesh.getElements(2, surf_tag)
    for etype, nodes in zip(etypes, enodes):
        if etype in (2, 9):
            nn = 3 if etype == 2 else 6
            conn = nodes.reshape(-1, nn)[:, :3].astype(int)
            return np.vectorize(tag_to_idx.get)(conn)
    return np.empty((0, 3), dtype=int)


def edges_from_pg(phys_id):
    """Boundary edge connectivity from a physical group."""
    edges = []
    for ent in gmsh.model.getEntitiesForPhysicalGroup(1, phys_id):
        etypes, _, enodes = gmsh.model.mesh.getElements(1, ent)
        for etype, nodes in zip(etypes, enodes):
            if etype == 1:
                edges.append(np.vectorize(tag_to_idx.get)(
                    nodes.reshape(-1, 2).astype(int)))
            elif etype == 8:
                edges.append(np.vectorize(tag_to_idx.get)(
                    nodes.reshape(-1, 3)[:, :2].astype(int)))
    if not edges:
        return np.empty((0, 2), dtype=int)
    return np.unique(np.sort(np.vstack(edges), axis=1), axis=0)


tris_tube_u = get_tris(s_tube_u)
tris_tube_l = get_tris(s_tube_l)
tris_paste  = get_tris(s_paste)
tris_mold   = get_tris(s_mold)

edges_dir = edges_from_pg(pg_dirichlet)
edges_rob = edges_from_pg(pg_robin)

gmsh.finalize()

# ================================================================
# 3) ACTIVE NODE REDUCTION
# ================================================================
used = set()
for arr in [tris_tube_u, tris_tube_l, tris_paste, tris_mold,
            edges_dir, edges_rob]:
    used |= set(arr.ravel())
used = np.array(sorted(used), dtype=int)

new_idx = -np.ones(coords_all.shape[0], dtype=int)
new_idx[used] = np.arange(len(used))

coords      = coords_all[used]
tris_tube_u = new_idx[tris_tube_u]
tris_tube_l = new_idx[tris_tube_l]
tris_paste  = new_idx[tris_paste]
tris_mold   = new_idx[tris_mold]
edges_dir   = new_idx[edges_dir]
edges_rob   = new_idx[edges_rob]

Nn = coords.shape[0]
tris_tube  = np.vstack([tris_tube_u, tris_tube_l])
triangles  = np.vstack([tris_tube, tris_paste, tris_mold])

print(f"\nMesh: {Nn} nodes, {len(triangles)} triangles")
print(f"  tube: {len(tris_tube)} ({len(tris_tube_u)} upper + {len(tris_tube_l)} lower)")
print(f"  paste: {len(tris_paste)}, mold: {len(tris_mold)}")
print(f"  Dirichlet edges: {len(edges_dir)}, Robin edges: {len(edges_rob)}")

# ================================================================
# 4) FEM ASSEMBLY — per-material stiffness + Robin BC
# ================================================================
f = np.zeros(Nn)
all_rows, all_cols, all_vals = [], [], []


def assemble_stiffness(tris, k_val):
    """Vectorized stiffness for P1 triangles with conductivity k_val."""
    if len(tris) == 0:
        return np.array([]), np.array([]), np.array([])
    i1, i2, i3 = tris[:, 0], tris[:, 1], tris[:, 2]
    x1, y1 = coords[i1, 0], coords[i1, 1]
    x2, y2 = coords[i2, 0], coords[i2, 1]
    x3, y3 = coords[i3, 0], coords[i3, 1]

    det  = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * np.abs(det)
    v    = area > 0

    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)[v]
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)[v]
    coeff = k_val / (4.0 * area[v])

    Ke = coeff[:, None, None] * (
        b[:, :, None] * b[:, None, :] +
        c[:, :, None] * c[:, None, :]
    )

    nd   = tris[v]
    rows = np.repeat(nd, 3, axis=1).ravel()
    cols = np.tile(nd, (1, 3)).ravel()
    return rows, cols, Ke.ravel()


for tri, k, name in [(tris_tube, k_tube,   "tube"),
                      (tris_paste, k_paste,  "paste"),
                      (tris_mold,  k_solder, "mold")]:
    r, c, v = assemble_stiffness(tri, k)
    all_rows.append(r); all_cols.append(c); all_vals.append(v)
    print(f"  assembled {name}: {len(tri)} elements, k={k}")

# Robin BC on all external convection edges
ei, ej = edges_rob[:, 0], edges_rob[:, 1]
dx = coords[ej, 0] - coords[ei, 0]
dy = coords[ej, 1] - coords[ei, 1]
Le = np.hypot(dx, dy)

fh = h_conv * Tinf * Le / 2.0
np.add.at(f, ei, fh)
np.add.at(f, ej, fh)

h_diag = h_conv * Le * (2.0 / 6.0)
h_off  = h_conv * Le * (1.0 / 6.0)
all_rows.append(np.concatenate([ei, ei, ej, ej]))
all_cols.append(np.concatenate([ei, ej, ei, ej]))
all_vals.append(np.concatenate([h_diag, h_off, h_off, h_diag]))

K = coo_matrix(
    (np.concatenate(all_vals),
     (np.concatenate(all_rows).astype(int),
      np.concatenate(all_cols).astype(int))),
    shape=(Nn, Nn)
).tocsr()

# ================================================================
# 5) DIRICHLET BC  +  SOLVE
# ================================================================
arc_nodes = np.unique(edges_dir.ravel())
free_mask = np.ones(Nn, dtype=bool)
free_mask[arc_nodes] = False
free_idx = np.where(free_mask)[0]
Nf, Nd = len(free_idx), len(arc_nodes)

print(f"\nDirichlet nodes: {Nd},  Free nodes: {Nf}")

K_ff   = K[free_idx, :][:, free_idx]
K_fd_g = K[free_idx, :][:, arc_nodes] @ np.full(Nd, T_wall)
rhs    = f[free_idx] - K_fd_g

T_free = spsolve(K_ff, rhs)

T = np.full(Nn, T_wall, dtype=float)
T[free_idx] = T_free

print(f"\nSolution: T_min = {T.min():.4f} C,  T_max = {T.max():.4f} C")

# Energy balance
q_in = np.sum(K[arc_nodes, :] @ T - f[arc_nodes])
Tm_rob = 0.5 * (T[edges_rob[:, 0]] + T[edges_rob[:, 1]])
Le_rob = np.hypot(coords[edges_rob[:, 1], 0] - coords[edges_rob[:, 0], 0],
                  coords[edges_rob[:, 1], 1] - coords[edges_rob[:, 0], 1])
q_out = np.sum(h_conv * (Tm_rob - Tinf) * Le_rob)

print(f"Heat in  (reaction): {q_in:.6e} W/m")
print(f"Heat out (Robin):    {q_out:.6e} W/m")
if abs(q_in) > 1e-14:
    print(f"Energy balance error: {abs(q_out - q_in) / abs(q_in):.2%}")

# ================================================================
# 6) PLOT — HALF DOMAIN
# ================================================================
triang_half = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)
pad = 0.00015

fig, ax = plt.subplots(figsize=(6.5, 6.0))
cf = ax.tricontourf(triang_half, T, levels=40, cmap="turbo")
cbar = fig.colorbar(cf, ax=ax, pad=0.03)
cbar.set_label("Temperature (°C)")
ax.triplot(triang_half, linewidth=0.08, alpha=0.12, color="k")

# Material interfaces
theta_u = np.linspace(0, np.pi / 2, 200)
theta_l = np.linspace(-np.pi / 2, 0, 200)
for r_val, lbl in [(R_tube, "tube/paste"), (R_paste, "paste/mold")]:
    ax.plot(xc + r_val * np.cos(theta_u), r_val * np.sin(theta_u),
            "w--", linewidth=0.8, alpha=0.7)
# Exposed tube outer arc (lower)
ax.plot(xc + R_tube * np.cos(theta_l), R_tube * np.sin(theta_l),
        "w--", linewidth=0.8, alpha=0.7)

ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Steady T — half domain (3-mat, T_wall={T_wall}°C)")
ax.grid(True, alpha=0.2)
ax.set_xlim(xc - pad / 3, W + pad)
ax.set_ylim(-R_tube - pad, H + pad)
plt.savefig(half_fig, dpi=300, bbox_inches="tight")
plt.show()

# ================================================================
# 7) MIRROR → FULL DOMAIN
# ================================================================
tol = 1e-12
on_sym  = np.isclose(coords[:, 0], xc, atol=tol)
N_half  = Nn
non_sym = ~on_sym
n_mirror = np.sum(non_sym)

coords_full = np.empty((N_half + n_mirror, 2))
coords_full[:N_half] = coords
coords_full[N_half:, 0] = 2.0 * xc - coords[non_sym, 0]
coords_full[N_half:, 1] = coords[non_sym, 1]

T_full = np.empty(N_half + n_mirror)
T_full[:N_half] = T
T_full[N_half:] = T[non_sym]

mirror_map = np.arange(N_half, dtype=int)
mirror_map[non_sym] = N_half + np.arange(n_mirror)

tri_m = np.column_stack([mirror_map[triangles[:, 0]],
                          mirror_map[triangles[:, 2]],
                          mirror_map[triangles[:, 1]]])
tris_full   = np.vstack([triangles, tri_m])
triang_full = mtri.Triangulation(coords_full[:, 0], coords_full[:, 1],
                                  tris_full)

fig, ax = plt.subplots(figsize=(7.0, 6.0))
cf = ax.tricontourf(triang_full, T_full, levels=40, cmap="turbo")
cbar = fig.colorbar(cf, ax=ax, pad=0.03)
cbar.set_label("Temperature (°C)")
ax.triplot(triang_full, linewidth=0.08, alpha=0.12, color="k")

# Full-domain material interfaces
theta_top = np.linspace(0, np.pi, 400)
theta_bot = np.linspace(-np.pi, 0, 400)
for r_val in [R_tube, R_paste]:
    ax.plot(xc + r_val * np.cos(theta_top), r_val * np.sin(theta_top),
            "w--", linewidth=0.8, alpha=0.7)
ax.plot(xc + R_tube * np.cos(theta_bot), R_tube * np.sin(theta_bot),
        "w--", linewidth=0.8, alpha=0.7)

ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Steady T — full domain (3-mat, T_wall={T_wall}°C)")
ax.grid(True, alpha=0.2)
ax.set_xlim(-pad, W + pad)
ax.set_ylim(-R_tube - pad, H + pad)
plt.savefig(full_fig, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved: {half_fig}, {full_fig}")
