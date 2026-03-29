# ================================================================
# Math 543 – BVP Project
# TRANSIENT 2D HEAT CONDUCTION (FEM) on Gmsh mesh (half-domain)
# + Full-domain visualization by symmetry mirroring
# + Animation: frames every 0.5 s from t=1 to t=20 (GIF)
#
# PDE:
#   rho*cp * dT/dt = div(k grad T)
#
# BCs:
#   ARC   : -k dT/dn = q_in                 (Neumann flux IN)
#   OUTER : -k dT/dn = h (T - Tinf)         (Robin convection)
#   SYM   : dT/dn = 0                       (symmetry)
#
# IC:
#   T(x,y,0) = T0
#
# Time integration:
#   Backward Euler (unconditionally stable)
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation

import gmsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized

# -------------------- PARAMETERS (EDIT HERE) ---------------------
prefix = "Transient"

# Geometry (mm -> m)
W_mm = 1.4
H_mm = 1.0
cut_diameter_mm = 1.0
R_mm = cut_diameter_mm / 2.0

# Material (stainless-ish)
k = 16.0            # W/m-K
rho = 8000.0        # kg/m^3
cp  = 500.0         # J/kg-K

# Convection
h = 15.0            # W/m^2-K
Tinf = 25.0         # °C

# Heat flux input on ARC
q_in = 1.0e3        # W/m^2

# Initial condition
T0 = 25.0           # °C

# Mesh sizing
h_global = 0.06e-3
h_arc    = 0.02e-3
dist_min = 0.04e-3
dist_max = 0.18e-3

# Time stepping
dt = 0.05           # s (solver internal step)
t_end = 500.0        # s

# Plot / animation settings
cmap = "turbo"
gif_name = "Temperature_transient_full_1to20_step0p5.gif"
gif_dpi = 200
gif_interval_ms = 200
levels = 50
# ---------------------------------------------------------------

# Unit conversions
W  = W_mm * 1e-3
H  = H_mm * 1e-3
R  = R_mm * 1e-3
xc = (W_mm / 2) * 1e-3

# Geometry validity checks
if R > (W - xc) + 1e-15:
    raise ValueError(f"Invalid geometry: need R <= W/2. Got R={R_mm} mm, W/2={W_mm/2} mm.")
if R > H + 1e-15:
    raise ValueError(f"Invalid geometry: need R <= H. Got R={R_mm} mm, H={H_mm} mm.")

print("=== TRANSIENT RUN (with animation capture) ===")
print(f"Geometry: W={W_mm}mm, H={H_mm}mm, cut diameter={cut_diameter_mm}mm (R={R_mm}mm)")
print(f"Material: k={k}, rho={rho}, cp={cp}")
print(f"BCs: q_in={q_in:.3e} W/m^2 on ARC, h={h} W/m^2K on OUTER, Tinf={Tinf}°C")
print(f"IC: T0={T0}°C")
print(f"Time: dt={dt}s, t_end={t_end}s")

# -------------------- BUILD MESH --------------------
# Avoid noisy finalize errors when already not initialized (Spyder reruns)
try:
    gmsh.finalize()
except Exception:
    pass

gmsh.initialize()
gmsh.model.add("transient_half_domain")

# Points (half-domain)
p1 = gmsh.model.geo.addPoint(xc + R, 0.0, 0.0, h_arc)     # bottom-arc intersection
p2 = gmsh.model.geo.addPoint(W,      0.0, 0.0, h_global)  # bottom-right
p3 = gmsh.model.geo.addPoint(W,      H,   0.0, h_global)  # top-right
p4 = gmsh.model.geo.addPoint(xc,     H,   0.0, h_global)  # top on symmetry
p5 = gmsh.model.geo.addPoint(xc,     R,   0.0, h_arc)     # arc-sym intersection
pc = gmsh.model.geo.addPoint(xc,     0.0, 0.0, h_arc)     # circle center

# Curves
l_bottom = gmsh.model.geo.addLine(p1, p2)          # OUTER
l_right  = gmsh.model.geo.addLine(p2, p3)          # OUTER
l_top    = gmsh.model.geo.addLine(p3, p4)          # OUTER
l_sym    = gmsh.model.geo.addLine(p4, p5)          # SYM
c_arc    = gmsh.model.geo.addCircleArc(p5, pc, p1) # ARC

# Surface
cloop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_sym, c_arc])
surf  = gmsh.model.geo.addPlaneSurface([cloop])
gmsh.model.geo.synchronize()

# Physical groups
pg_arc = gmsh.model.addPhysicalGroup(1, [c_arc])
pg_outer = gmsh.model.addPhysicalGroup(1, [l_bottom, l_right, l_top])
pg_sym = gmsh.model.addPhysicalGroup(1, [l_sym])
pg_dom = gmsh.model.addPhysicalGroup(2, [surf])

# Refinement near arc
field_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(field_dist, "CurvesList", [c_arc])
gmsh.model.mesh.field.setNumber(field_dist, "Sampling", 160)

field_thresh = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field_thresh, "InField", field_dist)
gmsh.model.mesh.field.setNumber(field_thresh, "SizeMin", h_arc)
gmsh.model.mesh.field.setNumber(field_thresh, "SizeMax", h_global)
gmsh.model.mesh.field.setNumber(field_thresh, "DistMin", dist_min)
gmsh.model.mesh.field.setNumber(field_thresh, "DistMax", dist_max)
gmsh.model.mesh.field.setAsBackgroundMesh(field_thresh)

gmsh.model.mesh.generate(2)

# -------------------- EXTRACT NODES / TRIANGLES / EDGES --------------------
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
coords_all = node_coords.reshape(-1, 3)[:, :2]
tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

# Triangles (domain)
elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
triangles_all = None
for etype, enodes in zip(elem_types, elem_node_tags):
    if etype in (2, 9):  # 2=linear tri, 9=quadratic tri
        nn = 3 if etype == 2 else 6
        conn = enodes.reshape(-1, nn)
        tri3 = conn[:, :3]
        triangles_all = np.vectorize(tag_to_idx.get)(tri3)
        break
if triangles_all is None:
    gmsh.finalize()
    raise RuntimeError("No triangle elements found in the domain.")

def edges_from_physical_group(phys_id):
    edges = []
    entities = gmsh.model.getEntitiesForPhysicalGroup(1, phys_id)
    for ent in entities:
        etypes, _, enodes = gmsh.model.mesh.getElements(1, ent)
        for etype, nodes in zip(etypes, enodes):
            if etype == 1:  # 2-node line
                conn = nodes.reshape(-1, 2)
                edges.append(np.vectorize(tag_to_idx.get)(conn))
            elif etype == 8:  # 3-node quadratic line -> endpoints
                conn = nodes.reshape(-1, 3)[:, :2]
                edges.append(np.vectorize(tag_to_idx.get)(conn))
    if not edges:
        return np.empty((0, 2), dtype=int)
    E = np.vstack(edges)
    E = np.sort(E, axis=1)
    return np.unique(E, axis=0)

edges_arc_all   = edges_from_physical_group(pg_arc)
edges_outer_all = edges_from_physical_group(pg_outer)
edges_sym_all   = edges_from_physical_group(pg_sym)

gmsh.finalize()

print(f"Mesh: nodes={coords_all.shape[0]}, tris={len(triangles_all)}")
print(f"Edges: ARC={len(edges_arc_all)}, OUTER={len(edges_outer_all)}, SYM={len(edges_sym_all)}")

# -------------------- ACTIVE NODE REDUCTION --------------------
used = set(triangles_all.ravel().tolist())
used |= set(edges_arc_all.ravel().tolist())
used |= set(edges_outer_all.ravel().tolist())
used |= set(edges_sym_all.ravel().tolist())

used = np.array(sorted(used), dtype=int)
new_index = -np.ones(coords_all.shape[0], dtype=int)
new_index[used] = np.arange(len(used))

coords = coords_all[used]
triangles = new_index[triangles_all]
edges_arc = new_index[edges_arc_all]
edges_outer = new_index[edges_outer_all]

Nn = coords.shape[0]
print(f"Active nodes used in FEM: {Nn}")

# -------------------- ASSEMBLE K, M, f --------------------
KI, KJ, KV = [], [], []
MI, MJ, MV = [], [], []
f = np.zeros(Nn)

# Elements
for tri in triangles:
    i1, i2, i3 = tri
    x1, y1 = coords[i1]
    x2, y2 = coords[i2]
    x3, y3 = coords[i3]

    det = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
    A = 0.5 * abs(det)
    if A <= 0:
        continue

    # Stiffness
    bvec = np.array([y2 - y3, y3 - y1, y1 - y2])
    cvec = np.array([x3 - x2, x1 - x3, x2 - x1])
    Ke = (k / (4.0 * A)) * (np.outer(bvec, bvec) + np.outer(cvec, cvec))

    # Mass (consistent)
    Me = (rho * cp) * (A / 12.0) * np.array([[2, 1, 1],
                                            [1, 2, 1],
                                            [1, 1, 2]])

    nodes = [i1, i2, i3]
    for a in range(3):
        for b_ in range(3):
            KI.append(nodes[a]); KJ.append(nodes[b_]); KV.append(Ke[a, b_])
            MI.append(nodes[a]); MJ.append(nodes[b_]); MV.append(Me[a, b_])

# Robin convection on OUTER: add to K and f
for (i, j) in edges_outer:
    L = np.hypot(*(coords[j] - coords[i]))

    Kh = (h * L / 6.0) * np.array([[2.0, 1.0],
                                  [1.0, 2.0]])
    fh = (h * Tinf * L / 2.0) * np.array([1.0, 1.0])

    f[i] += fh[0]; f[j] += fh[1]

    KI += [i, i, j, j]
    KJ += [i, j, i, j]
    KV += [Kh[0, 0], Kh[0, 1], Kh[1, 0], Kh[1, 1]]

# Neumann flux on ARC: add to f
for (i, j) in edges_arc:
    L = np.hypot(*(coords[j] - coords[i]))
    fq = (q_in * L / 2.0) * np.array([1.0, 1.0])
    f[i] += fq[0]; f[j] += fq[1]

K = coo_matrix((KV, (KI, KJ)), shape=(Nn, Nn)).tocsr()
M = coo_matrix((MV, (MI, MJ)), shape=(Nn, Nn)).tocsr()

# -------------------- TIME INTEGRATION (Backward Euler) --------------------
A_sys = (M / dt) + K
solve_A = factorized(A_sys.tocsc())

Tn = np.full(Nn, T0, dtype=float)

times = np.arange(0.0, t_end + 1e-12, dt)
tmax_hist = np.zeros_like(times)

# -------------------- ANIMATION FRAME SCHEDULE --------------------
# Desired output times:
# 0–2 s   → 0.01 s
# 2–10 s  → 0.2 s

times_anim_target = np.concatenate([
    np.arange(0.0, 2.0 + 1e-12, 0.01),
    np.arange(2.5, 500.0 + 1e-12, 0.5)
])

# Convert to solver indices (robust, integer-based)
anim_indices = sorted(set(int(round(t / dt)) for t in times_anim_target))
anim_indices_set = set(anim_indices)

T_history = []
times_anim = []

for n, t in enumerate(times):
    tmax_hist[n] = Tn.max()

    if n in anim_indices_set:
        T_history.append(Tn.copy())
        times_anim.append(t)

    if n == len(times) - 1:
        break

    rhs = f + (M / dt) @ Tn
    Tn = solve_A(rhs)

print("=== TRANSIENT DONE ===")
print(f"Final Tmin={Tn.min():.4f} °C, Tmax={Tn.max():.4f} °C")
print(f"Captured frames: {len(T_history)}")

# -------------------- BUILD FULL-DOMAIN TRIANGULATION (ONCE) ------------------
if len(T_history) == 0:
    raise RuntimeError("No frames captured for animation. Check time settings.")

tol = 1e-12
on_sym = np.isclose(coords[:, 0], xc, atol=tol)

N_half = coords.shape[0]
mirror_index = -np.ones(N_half, dtype=int)

coords_full = coords.copy()
for i in range(N_half):
    if on_sym[i]:
        mirror_index[i] = i
    else:
        mirror_index[i] = coords_full.shape[0]
        coords_full = np.vstack([coords_full, [2.0 * xc - coords[i, 0], coords[i, 1]]])

tri_m = np.column_stack([
    mirror_index[triangles[:, 0]],
    mirror_index[triangles[:, 2]],
    mirror_index[triangles[:, 1]],
])
triangles_full = np.vstack([triangles, tri_m])
triang_full = mtri.Triangulation(coords_full[:, 0], coords_full[:, 1], triangles_full)

def make_full_T(Thalf):
    Tfull = np.zeros(coords_full.shape[0])
    Tfull[:N_half] = Thalf
    for i in range(N_half):
        if not on_sym[i]:
            Tfull[mirror_index[i]] = Thalf[i]
    return Tfull

# Global vmin/vmax for stable colors
Tmin_global = min(Th.min() for Th in T_history)
Tmax_global = max(Th.max() for Th in T_history)

# -------------------- ANIMATION (FULL DOMAIN) --------------------
padx, pady = 0.00015, 0.00010

fig, ax = plt.subplots(figsize=(6.8, 5.6))

def draw_frame(Tfull, tnow):
    # Plot Delta-T for visibility
    Z = Tfull - Tinf

    vmin = float(np.min(Z))
    vmax = float(np.max(Z))
    # Prevent "single color" when range is extremely tiny
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1e-6

    cf = ax.tricontourf(
        triang_full, Z,
        levels=np.linspace(vmin, vmax, levels),
        cmap=cmap
    )

    # Contour lines help a LOT
    ax.tricontour(triang_full, Z, levels=10, linewidths=0.4, alpha=0.6)

    ax.set_aspect("equal")
    ax.set_xlim(-padx, W + padx)
    ax.set_ylim(-pady, H + pady)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)

    Tmin = float(np.min(Tfull))
    Tmax = float(np.max(Tfull))
    ax.set_title(f"ΔT = T - T∞  (full domain) — t = {tnow:.1f} s")

    # Put a small text box
    ax.text(
        0.02, 0.98,
        f"Tmin={Tmin:.2f}°C\nTmax={Tmax:.2f}°C\nΔ={Tmax-Tmin:.3f}°C",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
    )
    return cf

# Initial draw
Tfull0 = make_full_T(T_history[0])
cf0 = draw_frame(Tfull0, times_anim[0])
cbar = fig.colorbar(cf0, ax=ax, pad=0.03)
cbar.set_label("ΔT (°C)")

def update(frame):
    # Remove old contour artists (compatible with older Matplotlib)
    for coll in list(ax.collections):
        coll.remove()

    # Remove old contour line artists too
    for line in list(ax.lines):
        line.remove()

    # Remove old texts except keep axis labels/title handled by redraw
    # (We clear texts by removing all, then redraw)
    for txt in list(ax.texts):
        txt.remove()

    Tfull = make_full_T(T_history[frame])
    cf = draw_frame(Tfull, times_anim[frame])

    # Update colorbar to the new mappable
    cbar.update_normal(cf)
    return []

ani = animation.FuncAnimation(
    fig, update,
    frames=len(times_anim),
    interval=gif_interval_ms,
    blit=False
)

ani.save(gif_name, writer="pillow", dpi=gif_dpi)
plt.close(fig)

print(f"Animation saved as: {gif_name}")


# -------------------- Plot Tmax(t) --------------------
plt.figure(figsize=(6.5, 4.0))
plt.plot(times, tmax_hist)
plt.xlabel("Time (s)")
plt.ylabel("Max temperature (°C)")
plt.title("Tmax vs time (transient)")
plt.grid(True, alpha=0.3)
plt.savefig(f"{prefix}_Tmax_vs_time.png", dpi=300, bbox_inches="tight")
plt.show()
