# ================================================================
# Math 543 – BVP Project
# TRANSIENT 2D FEM — 3-MATERIAL MODEL — FULL TUBE
#
# Physical setup (half-domain, symmetry at x = W/2):
#
#   UPPER REGION (y > 0):  mold (rectangle)
#     Solder mold (60Sn/40Pb, k=50, rho=8500, cp=180)
#       └─ Thermal paste (annular, k=9, rho=2500, cp=800)
#            └─ Tube wall upper quarter (SS316, k=16, rho=8000, cp=500)
#                 └─ Water channel: T = T_wall  [Dirichlet]
#
#   LOWER REGION (y < 0):  exposed tube
#     Tube wall lower quarter (SS316, k=16, rho=8000, cp=500)
#       inner:  T = T_wall  [Dirichlet]
#       outer:  air convection  [Robin]
#
# PDE:   rho*cp * dT/dt = div(k grad T)
#
# BCs:
#   INNER ARC (full half-circle):    T = T_wall         [Dirichlet]
#   MOLD OUTER (bottom/right/top):   h(T-T_inf)         [Robin]
#   PASTE BOTTOM (y=0):              h(T-T_inf)         [Robin]
#   EXPOSED TUBE ARC (R_tube, y<0):  h(T-T_inf)         [Robin]
#   SYMMETRY (x = xc):               dT/dn = 0          [natural]
#
# Time integration:   Backward Euler (implicit, unconditionally stable)
# IC:                 T(x,y,0) = T0
#
# Outputs:
#   Twall_3mat_transient.gif
#   Twall_3mat_Tmax_vs_time.png
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation

import gmsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized

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

# Material properties (k, rho, cp)
k_tube   = 16.0;   rho_tube   = 8000.0;  cp_tube   = 500.0   # SS316
k_paste  = 9.0;    rho_paste  = 2500.0;  cp_paste  = 800.0   # thermal paste
k_solder = 50.0;   rho_solder = 8500.0;  cp_solder = 180.0   # 60Sn/40Pb

# Outer convection (air)
h_conv = 15.0      # W/m²K
Tinf   = 25.0      # °C

# Prescribed inner wall temperature
T_wall = 60.0      # °C

# Initial condition
T0 = 25.0          # °C  (= Tinf before heating starts)

# Mesh sizing (m)
h_global = 0.06e-3
h_arc    = 0.008e-3
h_mid    = 0.015e-3
dist_min = 0.02e-3
dist_max = 0.15e-3

# Time stepping
dt    = 0.05       # s
t_end = 500.0      # s

# Convergence criterion (auto-stop when max |T^{n+1} - T^n| < tol)
conv_tol = 1e-3    # C

# Animation capture schedule
t_dense_end   = 2.0     # s — dense capture up to here
dt_dense      = 0.01    # s
t_coarse_start = 2.5    # s
dt_coarse     = 0.5     # s

# Output
cmap         = "turbo"
gif_name     = "Twall_3mat_transient.gif"
gif_dpi      = 180
gif_interval = 120       # ms per frame
n_levels     = 50
plot_field   = "T"       # "T" or "dT"

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
print("TRANSIENT 2D — 3-material, full tube with exposed section")
print("=" * 60)
print(f"Tube: ID={tube_ID_mm}mm, OD={tube_OD_mm}mm")
print(f"Paste thickness: {paste_thickness_mm}mm")
print(f"Radii: R_in={R_in_mm}mm -> R_tube={R_tube_mm}mm -> R_paste={R_paste_mm}mm")
print(f"k: tube={k_tube}, paste={k_paste}, solder={k_solder} W/mK")
print(f"BC: T_wall={T_wall}C | h={h_conv} W/m2K, T_inf={Tinf}C | T0={T0}C")
print(f"Time: dt={dt}s, t_end={t_end}s")

# ================================================================
# 1) BUILD MESH  (gmsh)
# ================================================================
try:
    gmsh.finalize()
except Exception:
    pass

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("transient_3mat_full_tube")

# ---- Points ----
pc = gmsh.model.geo.addPoint(xc, 0.0, 0.0, h_arc)

p_in_top   = gmsh.model.geo.addPoint(xc,        R_in,  0.0, h_arc)
p_in_right = gmsh.model.geo.addPoint(xc + R_in, 0.0,   0.0, h_arc)
p_in_bot   = gmsh.model.geo.addPoint(xc,       -R_in,  0.0, h_arc)

p_tube_top   = gmsh.model.geo.addPoint(xc,          R_tube, 0.0, h_mid)
p_tube_right = gmsh.model.geo.addPoint(xc + R_tube, 0.0,    0.0, h_mid)
p_tube_bot   = gmsh.model.geo.addPoint(xc,         -R_tube, 0.0, h_mid)

p_paste_top   = gmsh.model.geo.addPoint(xc,           R_paste, 0.0, h_mid)
p_paste_right = gmsh.model.geo.addPoint(xc + R_paste, 0.0,     0.0, h_mid)

p_mold_br = gmsh.model.geo.addPoint(W,  0.0, 0.0, h_global)
p_mold_tr = gmsh.model.geo.addPoint(W,  H,   0.0, h_global)
p_mold_tl = gmsh.model.geo.addPoint(xc, H,   0.0, h_global)

# ---- Arcs ----
arc_in_u   = gmsh.model.geo.addCircleArc(p_in_top,    pc, p_in_right)
arc_in_l   = gmsh.model.geo.addCircleArc(p_in_right,  pc, p_in_bot)
arc_tube_u = gmsh.model.geo.addCircleArc(p_tube_top,  pc, p_tube_right)
arc_tube_l = gmsh.model.geo.addCircleArc(p_tube_right, pc, p_tube_bot)
arc_paste  = gmsh.model.geo.addCircleArc(p_paste_top, pc, p_paste_right)

# ---- Lines at y=0 ----
l_y0_tube  = gmsh.model.geo.addLine(p_in_right,    p_tube_right)
l_y0_paste = gmsh.model.geo.addLine(p_tube_right,  p_paste_right)
l_y0_mold  = gmsh.model.geo.addLine(p_paste_right, p_mold_br)

# ---- Mold edges ----
l_right = gmsh.model.geo.addLine(p_mold_br, p_mold_tr)
l_top   = gmsh.model.geo.addLine(p_mold_tr, p_mold_tl)

# ---- Symmetry lines ----
l_sym_mold   = gmsh.model.geo.addLine(p_mold_tl,  p_paste_top)
l_sym_paste  = gmsh.model.geo.addLine(p_paste_top, p_tube_top)
l_sym_tube_u = gmsh.model.geo.addLine(p_tube_top,  p_in_top)
l_sym_tube_l = gmsh.model.geo.addLine(p_tube_bot,  p_in_bot)

# ---- Surfaces ----
cl_tube_u = gmsh.model.geo.addCurveLoop([
    arc_in_u, l_y0_tube, -arc_tube_u, l_sym_tube_u])
s_tube_u = gmsh.model.geo.addPlaneSurface([cl_tube_u])

cl_tube_l = gmsh.model.geo.addCurveLoop([
    arc_tube_l, l_sym_tube_l, -arc_in_l, l_y0_tube])
s_tube_l = gmsh.model.geo.addPlaneSurface([cl_tube_l])

cl_paste = gmsh.model.geo.addCurveLoop([
    arc_tube_u, l_y0_paste, -arc_paste, l_sym_paste])
s_paste = gmsh.model.geo.addPlaneSurface([cl_paste])

cl_mold = gmsh.model.geo.addCurveLoop([
    arc_paste, l_y0_mold, l_right, l_top, l_sym_mold])
s_mold = gmsh.model.geo.addPlaneSurface([cl_mold])

gmsh.model.geo.synchronize()

# ---- Physical groups ----
pg_dirichlet = gmsh.model.addPhysicalGroup(1, [arc_in_u, arc_in_l])
pg_robin     = gmsh.model.addPhysicalGroup(1, [
    arc_tube_l, l_y0_paste, l_y0_mold, l_right, l_top])
pg_sym = gmsh.model.addPhysicalGroup(1, [
    l_sym_mold, l_sym_paste, l_sym_tube_u, l_sym_tube_l])

pg_s_tube_u = gmsh.model.addPhysicalGroup(2, [s_tube_u])
pg_s_tube_l = gmsh.model.addPhysicalGroup(2, [s_tube_l])
pg_s_paste  = gmsh.model.addPhysicalGroup(2, [s_paste])
pg_s_mold   = gmsh.model.addPhysicalGroup(2, [s_mold])

# ---- Mesh refinement ----
fd = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [
    arc_in_u, arc_in_l, arc_tube_u, arc_tube_l, arc_paste])
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
    etypes, _, enodes = gmsh.model.mesh.getElements(2, surf_tag)
    for etype, nodes in zip(etypes, enodes):
        if etype in (2, 9):
            nn = 3 if etype == 2 else 6
            conn = nodes.reshape(-1, nn)[:, :3].astype(int)
            return np.vectorize(tag_to_idx.get)(conn)
    return np.empty((0, 3), dtype=int)


def edges_from_pg(phys_id):
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

# ================================================================
# 4) FEM ASSEMBLY — K (stiffness) + M (mass) per material
# ================================================================
f_full = np.zeros(Nn)
K_rows, K_cols, K_vals = [], [], []
M_rows, M_cols, M_vals = [], [], []

M_template = np.array([[2, 1, 1],
                        [1, 2, 1],
                        [1, 1, 2]], dtype=float)


def assemble_KM(tris, k_val, rho_val, cp_val):
    """Vectorized stiffness + consistent mass for P1 triangles."""
    if len(tris) == 0:
        return (np.array([]),) * 6
    i1, i2, i3 = tris[:, 0], tris[:, 1], tris[:, 2]
    x1, y1 = coords[i1, 0], coords[i1, 1]
    x2, y2 = coords[i2, 0], coords[i2, 1]
    x3, y3 = coords[i3, 0], coords[i3, 1]

    det  = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * np.abs(det)
    v    = area > 0

    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)[v]
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)[v]

    # Stiffness
    coeff_K = k_val / (4.0 * area[v])
    Ke = coeff_K[:, None, None] * (
        b[:, :, None] * b[:, None, :] +
        c[:, :, None] * c[:, None, :]
    )

    # Consistent mass: rho*cp * (A/12) * [[2,1,1],[1,2,1],[1,1,2]]
    coeff_M = (rho_val * cp_val) * area[v] / 12.0
    Me = coeff_M[:, None, None] * M_template[None, :, :]

    nd   = tris[v]
    rows = np.repeat(nd, 3, axis=1).ravel()
    cols = np.tile(nd, (1, 3)).ravel()
    return rows, cols, Ke.ravel(), rows, cols, Me.ravel()


for tri, k, rho, cp, name in [
    (tris_tube, k_tube, rho_tube, cp_tube, "tube"),
    (tris_paste, k_paste, rho_paste, cp_paste, "paste"),
    (tris_mold, k_solder, rho_solder, cp_solder, "mold"),
]:
    kr, kc, kv, mr, mc, mv = assemble_KM(tri, k, rho, cp)
    K_rows.append(kr); K_cols.append(kc); K_vals.append(kv)
    M_rows.append(mr); M_cols.append(mc); M_vals.append(mv)
    print(f"  assembled {name}: {len(tri)} elements, k={k}, rho={rho}, cp={cp}")

# Robin BC on outer edges
ei, ej = edges_rob[:, 0], edges_rob[:, 1]
Le = np.hypot(coords[ej, 0] - coords[ei, 0], coords[ej, 1] - coords[ei, 1])

fh = h_conv * Tinf * Le / 2.0
np.add.at(f_full, ei, fh)
np.add.at(f_full, ej, fh)

h_diag = h_conv * Le * (2.0 / 6.0)
h_off  = h_conv * Le * (1.0 / 6.0)
robin_rows = np.concatenate([ei, ei, ej, ej])
robin_cols = np.concatenate([ei, ej, ei, ej])
robin_vals = np.concatenate([h_diag, h_off, h_off, h_diag])

# Build global K and M
def build_sparse(rows_list, cols_list, vals_list, extra_r=None, extra_c=None, extra_v=None):
    r = np.concatenate(rows_list + ([extra_r] if extra_r is not None else []))
    c = np.concatenate(cols_list + ([extra_c] if extra_c is not None else []))
    v = np.concatenate(vals_list + ([extra_v] if extra_v is not None else []))
    return coo_matrix((v, (r.astype(int), c.astype(int))), shape=(Nn, Nn)).tocsr()

K_full = build_sparse(K_rows, K_cols, K_vals, robin_rows, robin_cols, robin_vals)
M_full = build_sparse(M_rows, M_cols, M_vals)

# ================================================================
# 5) DIRICHLET SUBSPACE PARTITION
# ================================================================
arc_nodes = np.unique(edges_dir.ravel())
free_mask = np.ones(Nn, dtype=bool)
free_mask[arc_nodes] = False
free_idx = np.where(free_mask)[0]
Nf, Nd = len(free_idx), len(arc_nodes)

print(f"\nDirichlet nodes: {Nd},  Free nodes: {Nf}")

K_ff = K_full[free_idx, :][:, free_idx]
M_ff = M_full[free_idx, :][:, free_idx]

# Static RHS correction: -K_fd * T_wall (constant Dirichlet)
# M_fd terms cancel because T_wall is constant in time (see transient Twall notes).
K_fd_g = K_full[free_idx, :][:, arc_nodes] @ np.full(Nd, T_wall)
f_free_static = f_full[free_idx] - K_fd_g

# System matrix: (M_ff/dt + K_ff) — factorize once
A_ff = (M_ff / dt) + K_ff
solve_A = factorized(A_ff.tocsc())
print("System matrix factorized.")

# ================================================================
# 6) TIME STEPPING (Backward Euler) with convergence auto-stop
# ================================================================
Tn_free = np.full(Nf, T0, dtype=float)
n_max   = int(round(t_end / dt))

# Animation frame schedule (pre-compute target step indices)
times_anim_target = np.concatenate([
    np.arange(0.0, t_dense_end + 1e-12, dt_dense),
    np.arange(t_coarse_start, t_end + 1e-12, dt_coarse)
])
anim_set = set(int(round(t / dt)) for t in times_anim_target)
anim_set.add(0)  # always capture initial state

T_history   = []
times_anim  = []
tmax_list   = []
times_list  = []
converged   = False
conv_step   = -1

print(f"\nTime stepping: up to {n_max} steps, conv_tol={conv_tol} C ...")

for n in range(n_max + 1):
    t = n * dt

    # Reconstruct full T for recording
    Tn_full = np.full(Nn, T_wall, dtype=float)
    Tn_full[free_idx] = Tn_free
    tmax_list.append(Tn_full.max())
    times_list.append(t)

    # Capture animation frame
    if n in anim_set:
        T_history.append(Tn_free.copy())
        times_anim.append(t)

    # Progress print every 500 steps
    if n % 500 == 0:
        print(f"  step {n:>6d}  t={t:8.3f}s  Tmax={tmax_list[-1]:.6f}C")

    if n == n_max:
        break

    # Backward Euler step
    rhs = f_free_static + (M_ff / dt) @ Tn_free
    Tn_free_new = solve_A(rhs)

    # Convergence check: max absolute change in free DOFs
    dT_max = np.max(np.abs(Tn_free_new - Tn_free))

    Tn_free = Tn_free_new

    if dT_max < conv_tol and n > 0:
        # Converged! Capture the final state and stop
        conv_step = n + 1
        t_conv = (n + 1) * dt

        Tn_full = np.full(Nn, T_wall, dtype=float)
        Tn_full[free_idx] = Tn_free
        tmax_list.append(Tn_full.max())
        times_list.append(t_conv)

        T_history.append(Tn_free.copy())
        times_anim.append(t_conv)

        converged = True
        print(f"\n  >>> CONVERGED at step {conv_step}, t={t_conv:.3f}s")
        print(f"      max|dT| = {dT_max:.2e} < tol={conv_tol}")
        break

# Convert to arrays for plotting
tmax_hist = np.array(tmax_list)
times_arr = np.array(times_list)

# Final state
Tn_full = np.full(Nn, T_wall, dtype=float)
Tn_full[free_idx] = Tn_free

print(f"\nFinal: T_min={Tn_full.min():.4f}C, T_max={Tn_full.max():.4f}C")
if converged:
    print(f"Converged at t={times_arr[-1]:.3f}s (step {conv_step}/{n_max})")
else:
    print(f"Reached t_end={t_end}s without convergence (tol={conv_tol})")
print(f"Animation frames captured: {len(T_history)}")

# ================================================================
# 7) BUILD FULL-DOMAIN MIRROR
# ================================================================
tol = 1e-12
on_sym  = np.isclose(coords[:, 0], xc, atol=tol)
N_half  = Nn
non_sym = ~on_sym
n_mirror    = np.sum(non_sym)
non_sym_idx = np.where(non_sym)[0]

coords_full = np.empty((N_half + n_mirror, 2))
coords_full[:N_half] = coords
coords_full[N_half:, 0] = 2.0 * xc - coords[non_sym, 0]
coords_full[N_half:, 1] = coords[non_sym, 1]

mirror_map = np.arange(N_half, dtype=int)
mirror_map[non_sym] = N_half + np.arange(n_mirror)

tri_m = np.column_stack([mirror_map[triangles[:, 0]],
                          mirror_map[triangles[:, 2]],
                          mirror_map[triangles[:, 1]]])
triangles_full = np.vstack([triangles, tri_m])
triang_full = mtri.Triangulation(coords_full[:, 0], coords_full[:, 1],
                                  triangles_full)


def make_full_T(T_free_frame):
    """Reconstruct full-domain T from free-DOF snapshot."""
    Thalf = np.full(Nn, T_wall, dtype=float)
    Thalf[free_idx] = T_free_frame
    Tfull = np.empty(coords_full.shape[0])
    Tfull[:N_half] = Thalf
    Tfull[mirror_map[non_sym_idx]] = Thalf[non_sym_idx]
    return Tfull


# ================================================================
# 8) ANIMATION (adaptive per-frame colorbar)
# ================================================================
if plot_field == "dT":
    vals_all    = [make_full_T(Tf) - Tinf for Tf in T_history]
    field_label = "dT (°C)"
else:
    vals_all    = [make_full_T(Tf) for Tf in T_history]
    field_label = "T (°C)"

pad = 0.00015

fig, ax = plt.subplots(figsize=(7.0, 6.0))

Z0 = vals_all[0]
zmin0, zmax0 = float(np.min(Z0)), float(np.max(Z0))
if zmax0 - zmin0 < 1e-6:
    zmax0 = zmin0 + 1e-6
cf0 = ax.tricontourf(triang_full, Z0,
                      levels=np.linspace(zmin0, zmax0, n_levels), cmap=cmap)
cbar = fig.colorbar(cf0, ax=ax, pad=0.03)
cbar.set_label(field_label)

# Material interface lines
theta_top = np.linspace(0, np.pi, 400)
theta_bot = np.linspace(-np.pi, 0, 400)


def update(frame):
    for coll in list(ax.collections):
        coll.remove()
    for line in list(ax.lines):
        line.remove()
    for txt in list(ax.texts):
        txt.remove()

    Z = vals_all[frame]
    zmin_f = float(np.min(Z))
    zmax_f = float(np.max(Z))
    if (zmax_f - zmin_f) < 1e-6:
        mid = 0.5 * (zmin_f + zmax_f)
        zmin_f = mid - 0.5e-6
        zmax_f = mid + 0.5e-6

    lvl = np.linspace(zmin_f, zmax_f, n_levels)
    cf  = ax.tricontourf(triang_full, Z, levels=lvl, cmap=cmap)
    ax.tricontour(triang_full, Z, levels=10, linewidths=0.30,
                  alpha=0.6, colors="k")

    # Material interfaces
    for r_val in [R_tube, R_paste]:
        ax.plot(xc + r_val * np.cos(theta_top), r_val * np.sin(theta_top),
                "w--", lw=0.8, alpha=0.7)
    ax.plot(xc + R_tube * np.cos(theta_bot), R_tube * np.sin(theta_bot),
            "w--", lw=0.8, alpha=0.7)

    cbar.update_normal(cf)

    ax.set_aspect("equal")
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(-R_tube - pad, H + pad)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{field_label} — full domain  t = {times_anim[frame]:.2f} s  "
                 f"(T_wall={T_wall}°C)")
    ax.text(0.02, 0.98,
            f"min={zmin_f:.4f}\nmax={zmax_f:.4f}\ndelta={zmax_f - zmin_f:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    return []


ani = animation.FuncAnimation(fig, update, frames=len(T_history),
                               interval=gif_interval, blit=False)
ani.save(gif_name, writer="pillow", dpi=gif_dpi)
plt.close(fig)
print(f"Animation saved: {gif_name}")

# ================================================================
# 9) Tmax vs TIME
# ================================================================
plt.figure(figsize=(6.5, 4.0))
plt.plot(times_arr, tmax_hist, "b-", lw=1.2)
plt.xlabel("Time (s)")
plt.ylabel("Max temperature (°C)")
plt.title(f"Tmax vs time  (T_wall={T_wall}°C, T_inf={Tinf}°C)")
plt.axhline(T_wall, color="r", linestyle="--", alpha=0.6, label=f"T_wall={T_wall}°C")
plt.axhline(Tinf,   color="b", linestyle="--", alpha=0.6, label=f"T_inf={Tinf}°C")
if converged:
    plt.axvline(times_arr[-1], color="g", linestyle="-.", alpha=0.7,
                label=f"Converged t={times_arr[-1]:.1f}s")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Twall_3mat_Tmax_vs_time.png", dpi=300, bbox_inches="tight")
plt.show()

# ================================================================
# 10) FINAL STEADY-STATE CONTOUR (static plot of converged state)
# ================================================================
T_final_full = make_full_T(Tn_free)

fig, ax = plt.subplots(figsize=(7.0, 6.0))
cf = ax.tricontourf(triang_full, T_final_full, levels=40, cmap=cmap)
cbar_f = fig.colorbar(cf, ax=ax, pad=0.03)
cbar_f.set_label("Temperature (°C)")
ax.triplot(triang_full, linewidth=0.06, alpha=0.10, color="k")

for r_val in [R_tube, R_paste]:
    ax.plot(xc + r_val * np.cos(theta_top), r_val * np.sin(theta_top),
            "w--", lw=0.8, alpha=0.7)
ax.plot(xc + R_tube * np.cos(theta_bot), R_tube * np.sin(theta_bot),
        "w--", lw=0.8, alpha=0.7)

ax.set_aspect("equal")
ax.set_xlim(-pad, W + pad)
ax.set_ylim(-R_tube - pad, H + pad)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
status = f"CONVERGED t={times_arr[-1]:.1f}s" if converged else f"t={times_arr[-1]:.1f}s"
ax.set_title(f"Transient T — final state ({status}, T_wall={T_wall}°C)")
ax.grid(True, alpha=0.2)
plt.savefig("Twall_3mat_transient_final.png", dpi=300, bbox_inches="tight")
plt.show()
print(f"Final contour saved: Twall_3mat_transient_final.png")
