# ================================================================
# Math 543 – BVP Project
# TRANSIENT 2D HEAT CONDUCTION (FEM) — DIRICHLET INNER WALL
#
# Physical setup:
#   Microtube: OD=0.8 mm, ID=0.5 mm, stainless steel
#   The solid block sits on top of the tube.
#   Domain: tube wall + solid block (same material, combined).
#   Water fills the tube interior — inner wall temperature is PRESCRIBED.
#
# PDE:
#   rho*cp * dT/dt = div(k grad T)
#
# Boundary conditions:
#   INNER ARC (tube inner wall) : T = T_wall           [Dirichlet — constant]
#   OUTER surfaces              : -k dT/dn = h(T-Tinf)  [Robin]
#   SYMMETRY                    : dT/dn = 0              [natural]
#
# IC:
#   T(x,y,0) = T0
#
# Time integration:
#   Backward Euler (unconditionally stable)
#
# FEM approach (Dirichlet subspace partition):
#   DOFs split: free (f) vs Dirichlet (d).
#   System matrix: A_ff = M_ff/dt + K_ff   (assembled ONCE, factorized once)
#   At each step:  rhs = f_free - K_fd*T_wall + M_ff/dt * T_free^n
#                  T_free^{n+1} = A_ff^{-1} * rhs
#   Reconstruct:   T[free] = T_free,  T[dirichlet] = T_wall
#
# Outputs:
#   Twall_transient.gif
#   Twall_Tmax_vs_time.png
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation

import gmsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized

# -------------------- PARAMETERS (EDIT HERE) ---------------------
# Geometry (mm -> m)
W_mm = 1.4
H_mm = 1.0

# Tube geometry
tube_OD_mm = 0.8
tube_ID_mm = 0.5
R_inner_mm = tube_ID_mm / 2.0   # 0.25 mm

# Material: stainless steel
k   = 16.0     # W/m-K
rho = 8000.0   # kg/m^3
cp  = 500.0    # J/kg-K

# Outer convection
h    = 15.0    # W/m^2-K
Tinf = 25.0    # °C

# Prescribed inner wall temperature
T_wall = 60.0   # °C  (set from experiment / frontend)

# Initial condition
T0 = 25.0   # °C  (typically = Tinf before heating starts)

# Mesh sizing
h_global = 0.06e-3
h_arc    = 0.008e-3
dist_min = 0.02e-3
dist_max = 0.15e-3

# Time stepping
dt    = 0.05    # s
t_end = 500.0   # s

# Animation capture schedule
t_dense_end  = 2.0    # s  — dense capture up to here
dt_dense     = 0.01   # s
t_coarse_start = 2.5  # s
dt_coarse    = 0.5    # s

# Output
cmap = "turbo"
gif_name     = "Twall_transient.gif"
gif_dpi      = 180
gif_interval = 120   # ms per frame
n_levels     = 50
plot_field   = "T"   # "T" or "dT"
# ---------------------------------------------------------------

W  = W_mm * 1e-3
H  = H_mm * 1e-3
R  = R_inner_mm * 1e-3
xc = W / 2.0

if R > (W - xc) + 1e-15:
    raise ValueError("R_inner > W/2. Check dimensions.")
if R > H + 1e-15:
    raise ValueError("R_inner > H. Check dimensions.")

print("=== TRANSIENT 2D — Dirichlet inner wall ===")
print(f"Tube: OD={tube_OD_mm}mm, ID={tube_ID_mm}mm  =>  R_inner={R_inner_mm}mm")
print(f"Domain: W={W_mm}mm, H={H_mm}mm")
print(f"k={k}, rho={rho}, cp={cp}")
print(f"h={h}, Tinf={Tinf}°C, T_wall={T_wall}°C, T0={T0}°C")
print(f"dt={dt}s, t_end={t_end}s")

# -------------------- BUILD MESH --------------------
try:
    gmsh.finalize()
except Exception:
    pass

gmsh.initialize()
gmsh.model.add("transient_dirichlet_inner")

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

# -------------------- ASSEMBLE K, M, f — VECTORIZED --------------------
f_full = np.zeros(Nn)

# --- Vectorized element stiffness + mass (P1 triangles) ---
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
coeff_K = k / (4.0 * area[valid])

Ke_all = coeff_K[:, None, None] * (
    bvec[:, :, None] * bvec[:, None, :] +
    cvec[:, :, None] * cvec[:, None, :]
)  # (Nv, 3, 3)

# Consistent mass matrix: rho*cp*(A/12)*[[2,1,1],[1,2,1],[1,1,2]]
M_template = np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float)
coeff_M = (rho * cp) * area[valid] / 12.0
Me_all = coeff_M[:, None, None] * M_template[None, :, :]  # (Nv, 3, 3)

nd_valid = triangles[valid]  # (Nv, 3)
rows_el = np.repeat(nd_valid, 3, axis=1).ravel()
cols_el = np.tile(nd_valid, (1, 3)).ravel()

# --- Robin convection on OUTER edges ---
ei, ej = edges_outer[:, 0], edges_outer[:, 1]
dx = coords[ej, 0] - coords[ei, 0]
dy = coords[ej, 1] - coords[ei, 1]
Le = np.hypot(dx, dy)

fh_vals = h * Tinf * Le / 2.0
np.add.at(f_full, ei, fh_vals)
np.add.at(f_full, ej, fh_vals)

h_diag = h * Le * (2.0 / 6.0)
h_off  = h * Le * (1.0 / 6.0)
rows_R = np.concatenate([ei, ei, ej, ej])
cols_R = np.concatenate([ei, ej, ei, ej])
vals_R = np.concatenate([h_diag, h_off, h_off, h_diag])

# --- Build sparse K and M ---
all_rows = np.concatenate([rows_el, rows_R])
all_cols = np.concatenate([cols_el, cols_R])

K_full = coo_matrix((np.concatenate([Ke_all.ravel(), vals_R]),
                      (all_rows, all_cols)), shape=(Nn, Nn)).tocsr()
M_full = coo_matrix((np.concatenate([Me_all.ravel(), np.zeros(len(vals_R))]),
                      (all_rows, all_cols)), shape=(Nn, Nn)).tocsr()

# -------------------- DIRICHLET SUBSPACE PARTITION --------------------
arc_nodes = np.unique(edges_arc.ravel())
free_mask = np.ones(Nn, dtype=bool)
free_mask[arc_nodes] = False
free_idx = np.where(free_mask)[0]

Nf = len(free_idx)
Nd = len(arc_nodes)
print(f"Dirichlet nodes: {Nd},  Free nodes: {Nf}")

# Submatrices
K_ff = K_full[free_idx, :][:, free_idx]
M_ff = M_full[free_idx, :][:, free_idx]

# Static RHS correction: -K_fd * T_wall (constant since T_wall is constant)
K_fd_g = K_full[free_idx, :][:, arc_nodes] @ np.full(Nd, T_wall)
f_free_static = f_full[free_idx] - K_fd_g
# Note on M_fd cancellation:
#   Full partitioned BE:  (M_ff/dt + K_ff) T_f + (M_fd/dt + K_fd) T_d = f_f + M_ff/dt T_f^n + M_fd/dt T_d^n
#   Since T_d^{n+1} = T_d^n = T_wall (constant Dirichlet), the M_fd/dt terms
#   on LHS and RHS cancel exactly. Only -K_fd * T_wall remains as the static correction.
#   WARNING: If T_wall were time-varying, M_fd cross-coupling must be kept!

# System matrix for free DOFs (assembled ONCE)
A_ff = (M_ff / dt) + K_ff
solve_A = factorized(A_ff.tocsc())

# -------------------- INITIAL CONDITION --------------------
# T0 at all free nodes. Dirichlet nodes are set to T_wall from the first step.
# This creates a step-discontinuity at t=0 if T0 != T_wall.
# The Backward Euler scheme damps any resulting oscillations within ~2-3 steps,
# but if smoothness near t=0 matters, use a ramp:
#   for step n: T_wall_eff = T0 + (T_wall - T0) * min(1, n*dt / ramp_time)
Tn_free = np.full(Nf, T0, dtype=float)

times = np.arange(0.0, t_end + 1e-12, dt)

# -------------------- ANIMATION FRAME SCHEDULE --------------------
times_anim_target = np.concatenate([
    np.arange(0.0, t_dense_end + 1e-12, dt_dense),
    np.arange(t_coarse_start, t_end + 1e-12, dt_coarse)
])
anim_indices = sorted(set(int(round(t / dt)) for t in times_anim_target))
anim_set = set(anim_indices)

T_history   = []
times_anim  = []
tmax_hist   = np.zeros_like(times)

# -------------------- TIME LOOP --------------------
for n, t in enumerate(times):
    # Reconstruct full T for output
    Tn_full = np.full(Nn, T_wall, dtype=float)
    Tn_full[free_idx] = Tn_free
    tmax_hist[n] = Tn_full.max()

    if n in anim_set:
        T_history.append(Tn_free.copy())
        times_anim.append(t)

    if n == len(times) - 1:
        break

    # Backward Euler step (free DOFs only)
    rhs = f_free_static + (M_ff / dt) @ Tn_free
    Tn_free = solve_A(rhs)

# Final state
Tn_full = np.full(Nn, T_wall, dtype=float)
Tn_full[free_idx] = Tn_free

print("=== TRANSIENT DONE ===")
print(f"Final: Tmin={Tn_full.min():.4f} °C,  Tmax={Tn_full.max():.4f} °C")
print(f"Captured animation frames: {len(T_history)}")

# -------------------- BUILD FULL-DOMAIN MIRROR (vectorized) --------------------
tol = 1e-12
on_sym = np.isclose(coords[:, 0], xc, atol=tol)
N_half = coords.shape[0]
non_sym = ~on_sym
n_mirror = np.sum(non_sym)
non_sym_idx = np.where(non_sym)[0]

# Pre-allocate mirrored coords
coords_full = np.empty((N_half + n_mirror, 2))
coords_full[:N_half] = coords
coords_full[N_half:, 0] = 2.0 * xc - coords[non_sym, 0]
coords_full[N_half:, 1] = coords[non_sym, 1]

# Mirror index map
mirror_index = np.arange(N_half, dtype=int)
mirror_index[non_sym] = N_half + np.arange(n_mirror)

tri_m = np.column_stack([mirror_index[triangles[:, 0]],
                          mirror_index[triangles[:, 2]],
                          mirror_index[triangles[:, 1]]])
triangles_full = np.vstack([triangles, tri_m])
triang_full = mtri.Triangulation(coords_full[:, 0], coords_full[:, 1], triangles_full)

def make_full_T(T_free_frame):
    Tfull_half = np.full(Nn, T_wall, dtype=float)
    Tfull_half[free_idx] = T_free_frame
    Tfull = np.empty(coords_full.shape[0])
    Tfull[:N_half] = Tfull_half
    Tfull[mirror_index[non_sym_idx]] = Tfull_half[non_sym_idx]
    return Tfull

# -------------------- ANIMATION (per-frame adaptive contours) --------------------
# Because the system reaches steady state very fast (diffusion time ~ L^2/alpha),
# a fixed global colorbar would make later frames appear as a single color.
# Solution: each frame gets its own contour levels (adaptive), and a colorbar
# that updates to show the current range.

if plot_field == "dT":
    vals_all = [make_full_T(Tf) - Tinf for Tf in T_history]
    field_label = "ΔT (°C)"
else:
    vals_all = [make_full_T(Tf) for Tf in T_history]
    field_label = "T (°C)"

padx, pady = 0.00015, 0.00010

fig, ax = plt.subplots(figsize=(7.0, 5.6))

# Create a placeholder colorbar from first frame (will be updated per-frame)
Z0 = vals_all[0]
zmin0, zmax0 = float(np.min(Z0)), float(np.max(Z0))
if zmax0 - zmin0 < 1e-6:
    zmax0 = zmin0 + 1e-6
cf0 = ax.tricontourf(triang_full, Z0, levels=np.linspace(zmin0, zmax0, n_levels), cmap=cmap)
cbar = fig.colorbar(cf0, ax=ax, pad=0.03)
cbar.set_label(field_label)

def update(frame):
    for coll in list(ax.collections): coll.remove()
    for line in list(ax.lines):       line.remove()
    for txt  in list(ax.texts):       txt.remove()

    Z = vals_all[frame]
    zmin_f = float(np.min(Z))
    zmax_f = float(np.max(Z))

    # Ensure a visible range even when nearly uniform
    if (zmax_f - zmin_f) < 1e-6:
        mid = 0.5 * (zmin_f + zmax_f)
        zmin_f = mid - 0.5e-6
        zmax_f = mid + 0.5e-6

    lvl = np.linspace(zmin_f, zmax_f, n_levels)

    cf = ax.tricontourf(triang_full, Z, levels=lvl, cmap=cmap)
    ax.tricontour(triang_full, Z, levels=10, linewidths=0.30, alpha=0.6, colors="k")

    # Update colorbar to current frame's range
    cbar.update_normal(cf)

    ax.set_aspect("equal")
    ax.set_xlim(-padx, W + padx)
    ax.set_ylim(-pady, H + pady)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{field_label} — full domain  t = {times_anim[frame]:.2f} s  "
                 f"(T_wall={T_wall}°C)")
    ax.text(0.02, 0.98,
            f"min={zmin_f:.4f}\nmax={zmax_f:.4f}\nΔ={zmax_f - zmin_f:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    return []

ani = animation.FuncAnimation(fig, update, frames=len(T_history),
                               interval=gif_interval, blit=False)
ani.save(gif_name, writer="pillow", dpi=gif_dpi)
plt.close(fig)
print(f"Animation saved: {gif_name}")

# -------------------- Tmax vs TIME --------------------
plt.figure(figsize=(6.5, 4.0))
plt.plot(times, tmax_hist)
plt.xlabel("Time (s)")
plt.ylabel("Max temperature (°C)")
plt.title(f"Tmax vs time  (T_wall={T_wall}°C, Tinf={Tinf}°C)")
plt.axhline(T_wall, color="r", linestyle="--", alpha=0.6, label=f"T_wall={T_wall}°C")
plt.axhline(Tinf,   color="b", linestyle="--", alpha=0.6, label=f"Tinf={Tinf}°C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Twall_Tmax_vs_time.png", dpi=300, bbox_inches="tight")
plt.show()
