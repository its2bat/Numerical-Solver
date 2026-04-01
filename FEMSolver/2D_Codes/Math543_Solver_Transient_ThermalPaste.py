# ================================================================
# FEM Heat Transfer – BVP Project
# TRANSIENT 3D HEAT CONDUCTION (Linear Tetra FEM) with THERMAL PASTE
# Half-domain in x (symmetry at x=W/2) + full-domain mirror for viz
#
# PDE (piecewise materials):
#   rho*cp * dT/dt = div(k grad T)
#
# BCs:
#   INNER CYL (tube contact): -k dT/dn = q_in       (Neumann flux IN)
#   OUTER surfaces + Z-ends : -k dT/dn = h (T-Tinf) (Robin convection)
#   SYM plane (x=xc)        : dT/dn = 0             (natural Neumann)
#
# IC:
#   T(x,y,z,0) = T0
#
# Time integration:
#   Backward Euler: (M/dt + K) T^{n+1} = f + (M/dt) T^n
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import gmsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized

# -------------------- PARAMETERS (EDIT HERE) ---------------------
prefix = "Transient_3D_ThermalPaste"

# Geometry (mm -> m)
W_mm = 1.4
H_mm = 1.0
Lz_mm = 1.0               # extrusion length (z thickness)

cut_diameter_mm = 1.0
paste_thickness_mm = 0.1

# Materials
# Solid
k_s = 16.0
rho_s = 8000.0
cp_s  = 500.0

# Paste
k_p = 6.0
rho_p = 2500.0
cp_p  = 1000.0

# Convection
h = 15.0
Tinf = 25.0

# Heat flux on INNER CYL
q_in = 1.0e3  # W/m^2

# Initial condition
T0 = 25.0

# Mesh sizing (base)
h_global = 0.12e-3
h_inner  = 0.03e-3

# Inner refinement distances
dist_min_inner = 0.05e-3
dist_max_inner = 0.25e-3

# Side-surface refinement target
h_side_min = 0.04e-3
h_side_max = 0.14e-3
dist_min_side = 0.06e-3
dist_max_side = 0.45e-3

# Z direction refinement (IMPORTANT for side walls)
nz_layers = 8    # increase if side triangles still too big

# Time stepping
dt = 0.2
t_end = 50.0

# Animation capture schedule
t_dense_end = 2.0
dt_dense = 0.05
t_coarse_start = 2.0
dt_coarse = 0.25

# Visualization settings
plot_mode = "dT"     # "dT" or "T"
cmap_name = "turbo"

gif_name = "Transient3D_ThermalPaste_surface.gif"
gif_dpi = 160
gif_interval_ms = 80

# Mesh preview toggles (shown BEFORE solving)
show_mesh_half_boundary = True
show_mesh_full_boundary = True

# Temperature animation view
# Isometric-ish: azim=45, elev=25-35 is typical
view_elev = 28
view_azim = 45
# ---------------------------------------------------------------


# -------------------- UNIT CONVERSIONS -------------------------
W  = W_mm * 1e-3
H  = H_mm * 1e-3
Lz = Lz_mm * 1e-3

R_in = (cut_diameter_mm / 2.0) * 1e-3
t_p  = paste_thickness_mm * 1e-3
R_out = R_in + t_p

xc = (W_mm / 2.0) * 1e-3  # symmetry plane x=xc

if R_out > H + 1e-15:
    raise ValueError("Paste outer radius exceeds object height (H).")
if R_out > (W - xc) + 1e-15:
    raise ValueError("Paste outer radius exceeds half-width (W/2).")

print("=== TRANSIENT 3D RUN (thermal paste) ===")
print(f"W={W_mm}mm, H={H_mm}mm, Lz={Lz_mm}mm")
print(f"Cut d={cut_diameter_mm}mm => Rin={R_in*1e3:.3f}mm, paste t={paste_thickness_mm}mm => Rout={R_out*1e3:.3f}mm")
print(f"Solid: k={k_s}, rho={rho_s}, cp={cp_s}")
print(f"Paste: k={k_p}, rho={rho_p}, cp={cp_p}")
print(f"BCs: q_in={q_in:.3e} (inner cyl), h={h}, Tinf={Tinf}")
print(f"Time: dt={dt}, t_end={t_end}")
print(f"Plot: {plot_mode}")
print(f"Z layers (extrusion): {nz_layers}")

# ================================================================
# 1) BUILD 2D HALF CROSS-SECTION in (x,y), then EXTRUDE in z
# ================================================================
try:
    # avoid noisy errors; gmsh has isInitialized in recent versions
    if hasattr(gmsh, "isInitialized") and gmsh.isInitialized():
        gmsh.finalize()
    else:
        # older builds: finalize may throw, so keep it guarded
        try:
            gmsh.finalize()
        except Exception:
            pass
except Exception:
    pass

gmsh.initialize()
gmsh.model.add("paste3d_half")

# Points in 2D cross-section
p_in_bottom  = gmsh.model.geo.addPoint(xc + R_in,  0.0, 0.0, h_inner)
p_out_bottom = gmsh.model.geo.addPoint(xc + R_out, 0.0, 0.0, h_inner)

p_right_bot  = gmsh.model.geo.addPoint(W, 0.0, 0.0, h_global)
p_right_top  = gmsh.model.geo.addPoint(W, H,   0.0, h_global)
p_sym_top    = gmsh.model.geo.addPoint(xc, H,  0.0, h_global)

p_in_sym     = gmsh.model.geo.addPoint(xc, R_in,  0.0, h_inner)
p_out_sym    = gmsh.model.geo.addPoint(xc, R_out, 0.0, h_inner)

pc           = gmsh.model.geo.addPoint(xc, 0.0, 0.0, h_inner)

# Arcs
c_in  = gmsh.model.geo.addCircleArc(p_in_sym,  pc, p_in_bottom)   # inner arc
c_out = gmsh.model.geo.addCircleArc(p_out_sym, pc, p_out_bottom)  # outer arc (paste-solid interface)

# Lines
l_sym_solid = gmsh.model.geo.addLine(p_sym_top, p_out_sym)     # symmetry boundary (solid part)
l_sym_paste = gmsh.model.geo.addLine(p_out_sym, p_in_sym)      # interface segment

l_bot_solid = gmsh.model.geo.addLine(p_out_bottom, p_right_bot)  # outer bottom
l_bot_paste = gmsh.model.geo.addLine(p_in_bottom, p_out_bottom)  # radial interface segment

l_right = gmsh.model.geo.addLine(p_right_bot, p_right_top)
l_top   = gmsh.model.geo.addLine(p_right_top, p_sym_top)

gmsh.model.geo.synchronize()

# Paste surface
loop_paste = gmsh.model.geo.addCurveLoop([c_in, l_bot_paste, -c_out, l_sym_paste])
surf_paste = gmsh.model.geo.addPlaneSurface([loop_paste])

# Solid surface
loop_outer = gmsh.model.geo.addCurveLoop([l_bot_solid, l_right, l_top, l_sym_solid, c_out])
surf_solid = gmsh.model.geo.addPlaneSurface([loop_outer])

gmsh.model.geo.synchronize()

# Extrude to 3D volumes with multiple layers in z (fix coarse side walls!)
ext_paste = gmsh.model.geo.extrude([(2, surf_paste)], 0, 0, Lz,
                                   numElements=[nz_layers], recombine=False)
ext_solid = gmsh.model.geo.extrude([(2, surf_solid)], 0, 0, Lz,
                                   numElements=[nz_layers], recombine=False)
gmsh.model.geo.synchronize()

def pick_volume(extrude_out):
    vols = [tag for (dim, tag) in extrude_out if dim == 3]
    if not vols:
        raise RuntimeError("Extrusion did not create a volume.")
    return vols[0]

def pick_surfaces(extrude_out):
    return [tag for (dim, tag) in extrude_out if dim == 2]

vol_paste = pick_volume(ext_paste)
vol_solid = pick_volume(ext_solid)

surfs_paste_created = pick_surfaces(ext_paste)
surfs_solid_created = pick_surfaces(ext_solid)

# Mesh fields
# (a) refine near inner arc (tube contact)
f_dist_inner = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f_dist_inner, "CurvesList", [c_in])
gmsh.model.mesh.field.setNumber(f_dist_inner, "Sampling", 250)

f_th_inner = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f_th_inner, "InField", f_dist_inner)
gmsh.model.mesh.field.setNumber(f_th_inner, "SizeMin", h_inner)
gmsh.model.mesh.field.setNumber(f_th_inner, "SizeMax", h_global)
gmsh.model.mesh.field.setNumber(f_th_inner, "DistMin", dist_min_inner)
gmsh.model.mesh.field.setNumber(f_th_inner, "DistMax", dist_max_inner)

# (b) refine near side surfaces (reduce big triangles)
side_surface_candidates = list(set(surfs_paste_created + surfs_solid_created))

f_dist_side = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f_dist_side, "SurfacesList", side_surface_candidates)
gmsh.model.mesh.field.setNumber(f_dist_side, "Sampling", 200)

f_th_side = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f_th_side, "InField", f_dist_side)
gmsh.model.mesh.field.setNumber(f_th_side, "SizeMin", h_side_min)
gmsh.model.mesh.field.setNumber(f_th_side, "SizeMax", h_side_max)
gmsh.model.mesh.field.setNumber(f_th_side, "DistMin", dist_min_side)
gmsh.model.mesh.field.setNumber(f_th_side, "DistMax", dist_max_side)

# Combine fields (take minimum)
f_min = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_th_inner, f_th_side])
gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

# Global mesh options
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.option.setNumber("Mesh.Smoothing", 3)

gmsh.model.mesh.generate(3)

# ================================================================
# 2) EXTRACT NODES & ELEMENTS
# ================================================================
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
coords_all = node_coords.reshape(-1, 3)  # (x,y,z)
tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

print(f"Mesh total nodes (gmsh): {coords_all.shape[0]}")

def get_tets_from_volume(vol_tag):
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3, vol_tag)
    for etype, enodes in zip(elem_types, elem_node_tags):
        if etype in (4, 11):
            nn = 4 if etype == 4 else 10
            conn = enodes.reshape(-1, nn)[:, :4]
            return np.vectorize(tag_to_idx.get)(conn)
    return None

tets_solid_all = get_tets_from_volume(vol_solid)
tets_paste_all = get_tets_from_volume(vol_paste)
if tets_solid_all is None or tets_paste_all is None:
    gmsh.finalize()
    raise RuntimeError("Failed to extract tetrahedra. Check 3D meshing.")

print(f"Tetrahedra: solid={len(tets_solid_all)}, paste={len(tets_paste_all)}")

# Collect boundary surfaces of both volumes and extract boundary triangles
boundary_surfs = set()
for v in [vol_solid, vol_paste]:
    bs = gmsh.model.getBoundary([(3, v)], oriented=False, recursive=False)
    for (dim, tag) in bs:
        if dim == 2:
            boundary_surfs.add(tag)
boundary_surfs = sorted(boundary_surfs)

def get_tris_from_surface(surf_tag):
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, surf_tag)
    for etype, enodes in zip(elem_types, elem_node_tags):
        if etype in (2, 9):
            nn = 3 if etype == 2 else 6
            conn = enodes.reshape(-1, nn)[:, :3]
            return np.vectorize(tag_to_idx.get)(conn)
    return np.empty((0, 3), dtype=int)

all_bdry_tris = []
for s in boundary_surfs:
    tris = get_tris_from_surface(s)
    if len(tris) > 0:
        all_bdry_tris.append(tris)
if not all_bdry_tris:
    gmsh.finalize()
    raise RuntimeError("No boundary triangles found.")
all_bdry_tris = np.vstack(all_bdry_tris)

# Keep gmsh alive only until we are done extracting
gmsh.finalize()

# ================================================================
# 3) ACTIVE NODE REDUCTION
# ================================================================
used = set(tets_solid_all.ravel().tolist()) | set(tets_paste_all.ravel().tolist())
used |= set(all_bdry_tris.ravel().tolist())

used = np.array(sorted(used), dtype=int)
new_index = -np.ones(coords_all.shape[0], dtype=int)
new_index[used] = np.arange(len(used))

coords = coords_all[used]
tets_solid = new_index[tets_solid_all]
tets_paste = new_index[tets_paste_all]
bdry_tris  = new_index[all_bdry_tris]

Nn = coords.shape[0]
print(f"Active nodes used in FEM: {Nn}")

# ================================================================
# 4) CLASSIFY BC TRIANGLES
# ================================================================
def tri_is_inner(tri):
    pts = coords[tri]
    x = pts[:, 0]; y = pts[:, 1]
    r = np.sqrt((x - xc)**2 + y**2)
    return np.all(np.abs(r - R_in) < 3e-6)

def tri_is_sym(tri):
    pts = coords[tri]
    return np.all(np.abs(pts[:, 0] - xc) < 3e-10)

inner_tris = []
sym_tris = []
outer_tris = []

for tri in bdry_tris:
    if tri_is_sym(tri):
        sym_tris.append(tri)
    elif tri_is_inner(tri):
        inner_tris.append(tri)
    else:
        outer_tris.append(tri)

inner_tris = np.array(inner_tris, dtype=int)
sym_tris   = np.array(sym_tris, dtype=int)
outer_tris = np.array(outer_tris, dtype=int)

print("Boundary triangles classified:")
print(f"  inner (flux): {len(inner_tris)}")
print(f"  symmetry     : {len(sym_tris)}")
print(f"  outer (conv) : {len(outer_tris)}")

# ================================================================
# 5) SHOW MESH FIRST (BEFORE SOLVING) — HALF + FULL
# ================================================================
def mirror_full_geometry(coords_half, tris_half):
    on_sym = np.isclose(coords_half[:, 0], xc, atol=1e-12)
    N_half = coords_half.shape[0]
    mirror_index = -np.ones(N_half, dtype=int)

    coords_full = coords_half.copy()
    for i in range(N_half):
        if on_sym[i]:
            mirror_index[i] = i
        else:
            mirror_index[i] = coords_full.shape[0]
            coords_full = np.vstack([coords_full, [2.0*xc - coords_half[i,0],
                                                  coords_half[i,1],
                                                  coords_half[i,2]]])

    tris_m = np.column_stack([
        mirror_index[tris_half[:, 0]],
        mirror_index[tris_half[:, 2]],
        mirror_index[tris_half[:, 1]],
    ])
    tris_full = np.vstack([tris_half, tris_m])
    return coords_full, tris_full

if show_mesh_half_boundary or show_mesh_full_boundary:
    # half-domain boundary mesh
    if show_mesh_half_boundary:
        fig = plt.figure(figsize=(7.4, 5.8))
        ax = fig.add_subplot(111, projection="3d")
        P = coords
        TRI = bdry_tris

        ax.plot_trisurf(
            P[:,0], P[:,1], P[:,2],
            triangles=TRI,
            linewidth=0.15,
            antialiased=True,
            alpha=0.40,
            shade=False
        )
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_title("Boundary Mesh — HALF domain (shown BEFORE solving)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, Lz)
        try:
            ax.set_box_aspect((W, H, Lz))
        except Exception:
            pass
        plt.tight_layout()
        plt.show()

    # full-domain boundary mesh (mirrored)
    if show_mesh_full_boundary:
        Pfull, TRIfull = mirror_full_geometry(coords, bdry_tris)

        fig = plt.figure(figsize=(7.8, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            Pfull[:,0], Pfull[:,1], Pfull[:,2],
            triangles=TRIfull,
            linewidth=0.12,
            antialiased=True,
            alpha=0.35,
            shade=False
        )
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_title("Boundary Mesh — FULL domain (mirrored) — BEFORE solving")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, Lz)
        try:
            ax.set_box_aspect((W, H, Lz))
        except Exception:
            pass
        plt.tight_layout()
        plt.show()

# ================================================================
# 6) ASSEMBLE 3D FEM MATRICES (linear tetra)
# ================================================================
KI, KJ, KV = [], [], []
MI, MJ, MV = [], [], []
F = np.zeros(Nn)

def tet_gradients_and_volume(x):
    X = np.ones((4, 4))
    X[:, 1:] = x
    detX = np.linalg.det(X)
    V = abs(detX) / 6.0
    if V <= 0:
        return None, None
    invX = np.linalg.inv(X)
    grads = invX[1:4, :].T  # (4,3)
    return grads, V

def assemble_tets(tets, k_mat, rho_mat, cp_mat):
    Me_template = np.array([
        [2,1,1,1],
        [1,2,1,1],
        [1,1,2,1],
        [1,1,1,2]
    ], dtype=float)
    for tet in tets:
        pts = coords[tet]
        grads, V = tet_gradients_and_volume(pts)
        if grads is None:
            continue
        Ke = k_mat * V * (grads @ grads.T)
        Me = (rho_mat * cp_mat) * (V / 20.0) * Me_template

        for a in range(4):
            ia = tet[a]
            for b in range(4):
                ib = tet[b]
                KI.append(ia); KJ.append(ib); KV.append(Ke[a, b])
                MI.append(ia); MJ.append(ib); MV.append(Me[a, b])

assemble_tets(tets_solid, k_s, rho_s, cp_s)
assemble_tets(tets_paste, k_p, rho_p, cp_p)

tri_M = np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float)

def tri_area(p1, p2, p3):
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

# Robin convection on OUTER surfaces
for tri in outer_tris:
    i, j, k = tri
    p1, p2, p3 = coords[i], coords[j], coords[k]
    A = tri_area(p1, p2, p3)
    if A <= 0:
        continue
    Kh = h * (A / 12.0) * tri_M
    fh = h * Tinf * (A / 3.0) * np.ones(3)

    idx = [i, j, k]
    for a in range(3):
        F[idx[a]] += fh[a]
        for b in range(3):
            KI.append(idx[a]); KJ.append(idx[b]); KV.append(Kh[a, b])

# Neumann flux on INNER cylinder
for tri in inner_tris:
    i, j, k = tri
    p1, p2, p3 = coords[i], coords[j], coords[k]
    A = tri_area(p1, p2, p3)
    if A <= 0:
        continue
    fq = q_in * (A / 3.0) * np.ones(3)
    F[i] += fq[0]; F[j] += fq[1]; F[k] += fq[2]

K = coo_matrix((KV, (KI, KJ)), shape=(Nn, Nn)).tocsr()
M = coo_matrix((MV, (MI, MJ)), shape=(Nn, Nn)).tocsr()

# ================================================================
# 7) TRANSIENT (Backward Euler)
# ================================================================
A_sys = (M / dt) + K
solve_A = factorized(A_sys.tocsc())

Tn = np.full(Nn, T0, dtype=float)
times = np.arange(0.0, t_end + 1e-12, dt)

times_anim_target = np.concatenate([
    np.arange(0.0, t_dense_end + 1e-12, dt_dense),
    np.arange(t_coarse_start, t_end + 1e-12, dt_coarse)
])
anim_indices = sorted(set(int(round(t / dt)) for t in times_anim_target))
anim_set = set(anim_indices)

T_history = []
times_anim = []

for n, t in enumerate(times):
    if n in anim_set:
        T_history.append(Tn.copy())
        times_anim.append(t)

    if n == len(times) - 1:
        break

    rhs = F + (M / dt) @ Tn
    Tn = solve_A(rhs)

print("=== TRANSIENT DONE ===")
print(f"Final Tmin={Tn.min():.4f} °C, Tmax={Tn.max():.4f} °C")
print(f"Captured frames: {len(T_history)}")

if len(T_history) == 0:
    raise RuntimeError("No frames captured. Check animation schedule.")

# ================================================================
# 8) FULL-DOMAIN MIRROR (x -> 2xc - x)
# ================================================================
on_sym = np.isclose(coords[:, 0], xc, atol=1e-12)
N_half = coords.shape[0]
mirror_index = -np.ones(N_half, dtype=int)

coords_full = coords.copy()
for i in range(N_half):
    if on_sym[i]:
        mirror_index[i] = i
    else:
        mirror_index[i] = coords_full.shape[0]
        coords_full = np.vstack([coords_full, [2.0*xc - coords[i,0], coords[i,1], coords[i,2]]])

def make_full_T(Thalf):
    Tfull = np.zeros(coords_full.shape[0])
    Tfull[:N_half] = Thalf
    for i in range(N_half):
        if not on_sym[i]:
            Tfull[mirror_index[i]] = Thalf[i]
    return Tfull

bdry_m = np.column_stack([
    mirror_index[bdry_tris[:, 0]],
    mirror_index[bdry_tris[:, 2]],
    mirror_index[bdry_tris[:, 1]],
])
bdry_tris_full = np.vstack([bdry_tris, bdry_m])

# ================================================================
# 9) ANIMATION (SURFACE ΔT or T) — stable global scale (no solid colors)
# ================================================================
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import Normalize

def field_from_T(T):
    return (T - Tinf) if plot_mode.lower() == "dt" else T

# Build a GLOBAL range like your 2D method, but make it robust:
# - use min/max across frames
# - if too small, expand by a minimum visible span
Zmins, Zmaxs = [], []
for Th in T_history:
    Tf = make_full_T(Th)
    Z = field_from_T(Tf)
    Zmins.append(float(Z.min()))
    Zmaxs.append(float(Z.max()))

Zmin = float(np.min(Zmins))
Zmax = float(np.max(Zmaxs))

# Minimum visible span to avoid "one color" effect (tune if you want)
MIN_SPAN = 0.5 if plot_mode.lower() == "dt" else 0.5  # degrees
if (Zmax - Zmin) < MIN_SPAN:
    mid = 0.5 * (Zmin + Zmax)
    Zmin = mid - 0.5 * MIN_SPAN
    Zmax = mid + 0.5 * MIN_SPAN

norm = Normalize(vmin=Zmin, vmax=Zmax)
cmap_obj = mpl.colormaps.get_cmap(cmap_name)

P = coords_full
TRI = bdry_tris_full

fig = plt.figure(figsize=(9.0, 6.8))
ax = fig.add_subplot(111, projection="3d")

label = "ΔT (°C)" if plot_mode.lower() == "dt" else "T (°C)"
mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.78)
cbar.set_label(label)

# IMPORTANT: create trisurf with NO edges (so mesh doesn't appear on temp map)
surf = ax.plot_trisurf(
    P[:, 0], P[:, 1], P[:, 2],
    triangles=TRI,
    linewidth=0.0,
    antialiased=False,
    shade=False
)
# kill edges explicitly (prevents faint mesh lines on some backends)
try:
    surf.set_edgecolor("none")
except Exception:
    pass

# isometric-ish view
ax.view_init(elev=view_elev, azim=view_azim)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_zlim(0, Lz)
try:
    ax.set_box_aspect((W, H, Lz))
except Exception:
    pass

def update(frame):
    Tfull = make_full_T(T_history[frame])
    Z = field_from_T(Tfull)

    # face scalar = mean of vertex scalar for each triangle
    Zf = Z[TRI].mean(axis=1)
    surf.set_facecolors(cmap_obj(norm(Zf)))

    ax.set_title(f"{label} on boundary surface — t = {times_anim[frame]:.2f} s")
    return (surf,)

ani = animation.FuncAnimation(
    fig, update,
    frames=len(times_anim),
    interval=gif_interval_ms,
    blit=False
)

ani.save(gif_name, writer="pillow", dpi=gif_dpi)
plt.close(fig)

print(f"Animation saved as: {gif_name}")
print(f"Global color range used: [{Zmin:.3f}, {Zmax:.3f}] ({label})")
