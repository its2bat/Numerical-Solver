# ================================================================
# FEM Heat Transfer – BVP Project
# TRANSIENT 3D HEAT CONDUCTION (FEM) with THERMAL PASTE LAYER
# FULL mirrored model + interactive CLIP + SLICE on X/Y/Z via PyVista
#
# Window order:
#   1) Mesh preview (close -> continues)
#   2) Whole-object temperature (T) preview (close -> continues)
#   3) Interactive slicer view:
#        - Body colored by T with T scalar bar
#        - Slices colored by dT with ONE dT scalar bar (total 2 bars)
#
# PDE (transient):
#   rho c dT/dt - div(k grad T) = 0
#
# BCs:
#   INNER cylinder:  -k dT/dn = q_in(t)        (Neumann flux IN)
#   OUTER boundary:  -k dT/dn = h (T-Tinf)     (Robin convection)
#   SYM plane x=xc:   dT/dn = 0                (natural Neumann)
#
# Time integration (choose):
#   - Backward Euler (BE):  (M/dt + K) T^{n+1} = f^{n+1} + (M/dt) T^n
#   - Crank–Nicolson (CN):  (M/dt + 0.5 K) T^{n+1} = (M/dt - 0.5 K) T^n + 0.5(f^{n+1}+f^n)
#
# Units:
#   - Solving is done in SI (meters)
#   - Visualization is in mm
# ================================================================

import numpy as np
import gmsh
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# -------------------- PARAMETERS ---------------------
# Geometry in mm (converted to meters internally for FEM)
W_mm = 1.4
H_mm = 1.0
Lz_mm = 1.0

cut_diameter_mm = 1.0
paste_thickness_mm = 0.1

# Materials (conductivity)
k_s = 16.0   # W/m-K (solid)
k_p = 6.0    # W/m-K (paste)

# Materials (transient: rho, cp)
# Choose realistic values for your materials
rho_s = 8000.0   # kg/m^3
cp_s  = 500.0    # J/kg-K
rho_p = 2500.0   # kg/m^3
cp_p  = 800.0    # J/kg-K

# Convection
h = 15.0     # W/m^2-K
Tinf = 25.0  # °C

# Heat flux on inner cylinder (can be time-varying)
q_in0 = 1.0e3  # W/m^2 baseline amplitude

# Time settings
time_scheme = "BE"   # "BE" (Backward Euler) or "CN" (Crank–Nicolson)
dt = 0.5            # seconds
t_end = 1000.0          # seconds
T0 = Tinf            # initial temperature everywhere (°C)

# OPTIONAL flux ramp for stability/realism (CFD-like)
# ramp_time = 0.0 means step input (instant q_in0).
ramp_time = 0.5      # seconds (set 0.0 to disable)

# Mesh sizing (meters)
h_global = 0.06e-3
h_inner  = 0.02e-3

# z extrusion layers
nz_layers = 28

# Curvature sizing
curv_refine = 25

# Save final VTU (FULL, meters)
save_vtu_path = r"D:\Me543\3D_Sections\Transient3D_ThermalPaste_full.vtu"

# Visualization
cmap_name = "turbo"
SBAR_FMT = "%.3g"

# Optional display rotations
apply_rotation_preview = True
apply_rotation_gui = False
rot_xy_deg = 10.0
rot_yz_deg = -90.0

# Clip convention
keep_side = "ge"  # ge keeps >= slider, le keeps <= slider

# Auto clim settings
USE_PERCENTILE_CLIM = True
PCT_LO, PCT_HI = 1.0, 99.0
MIN_SPAN = 1e-6
PAD_FRAC = 0.02
# -----------------------------------------------------


# -------------------- UNIT CONVERSIONS ---------------------
W  = W_mm  * 1e-3
H  = H_mm  * 1e-3
Lz = Lz_mm * 1e-3

R_in  = (cut_diameter_mm / 2.0) * 1e-3
t_p   = paste_thickness_mm * 1e-3
R_out = R_in + t_p

xc = W / 2.0  # symmetry plane

if R_out > H + 1e-15:
    raise ValueError("Paste outer radius exceeds H. Reduce paste thickness or increase H.")
if R_out > (W - xc) + 1e-15:
    raise ValueError("Paste outer radius exceeds half-width. Reduce paste thickness or increase W.")

rc_s = rho_s * cp_s  # volumetric heat capacity (J/m^3-K)
rc_p = rho_p * cp_p

print("=== TRANSIENT 3D RUN (thermal paste) ===")
print(f"W={W_mm}mm, H={H_mm}mm, Lz={Lz_mm}mm")
print(f"Rin={R_in*1e3:.3f}mm, Rout={R_out*1e3:.3f}mm")
print(f"k_s={k_s}, k_p={k_p}, h={h}, Tinf={Tinf}, q_in0={float(q_in0):.3e}")
print(f"rho_s,cp_s={rho_s},{cp_s} => rc_s={rc_s:.3e}  |  rho_p,cp_p={rho_p},{cp_p} => rc_p={rc_p:.3e}")
print(f"time: scheme={time_scheme}, dt={dt}, t_end={t_end}, ramp_time={ramp_time}")


def safe_finalize():
    try:
        if hasattr(gmsh, "isInitialized"):
            if gmsh.isInitialized():
                gmsh.finalize()
        else:
            gmsh.finalize()
    except Exception:
        pass


# -------------------- BUILD + MESH (Gmsh OCC) ---------------------
safe_finalize()
gmsh.initialize()
gmsh.model.add("transient3d_paste_occ")

# Half rectangle: x in [xc, W], y in [0, H]
rect = gmsh.model.occ.addRectangle(xc, 0.0, 0.0, W - xc, H)

# Disks (full), intersect with rect -> quarter
disk_out = gmsh.model.occ.addDisk(xc, 0.0, 0.0, R_out, R_out)
disk_in  = gmsh.model.occ.addDisk(xc, 0.0, 0.0, R_in,  R_in)

gmsh.model.occ.synchronize()

q_out, _ = gmsh.model.occ.intersect([(2, rect)], [(2, disk_out)], removeObject=False, removeTool=False)
q_in_, _ = gmsh.model.occ.intersect([(2, rect)], [(2, disk_in )], removeObject=False, removeTool=False)

if len(q_out) == 0 or len(q_in_) == 0:
    safe_finalize()
    raise RuntimeError("OCC intersect failed.")

paste_cut, _ = gmsh.model.occ.cut(q_out, q_in_, removeObject=False, removeTool=False)
if len(paste_cut) == 0:
    safe_finalize()
    raise RuntimeError("OCC cut failed for paste surface.")

solid_cut, _ = gmsh.model.occ.cut([(2, rect)], q_out, removeObject=False, removeTool=False)
if len(solid_cut) == 0:
    safe_finalize()
    raise RuntimeError("OCC cut failed for solid surface.")

paste_faces = [e for e in paste_cut if e[0] == 2]
solid_faces = [e for e in solid_cut if e[0] == 2]
if not paste_faces or not solid_faces:
    safe_finalize()
    raise RuntimeError("Paste/Solid face extraction failed.")

gmsh.model.occ.synchronize()

# Extrude both together
ext = gmsh.model.occ.extrude(solid_faces + paste_faces, 0, 0, Lz, numElements=[nz_layers], recombine=False)
gmsh.model.occ.synchronize()

vols = [e for e in ext if e[0] == 3]
if len(vols) < 2:
    safe_finalize()
    raise RuntimeError("Expected >=2 volumes after extrusion.")
vol_tags = [v[1] for v in vols]

# Mesh options
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv_refine)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_inner)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_global)

gmsh.model.mesh.generate(3)

# Extract nodes
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
P_all = node_coords.reshape(-1, 3)
tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

# Extract tets by volume
def get_tets_from_volume(vol_tag):
    etypes, _, enodes = gmsh.model.mesh.getElements(3, vol_tag)
    for etype, nodes in zip(etypes, enodes):
        if etype in (4, 11):  # linear tet / quadratic tet
            nn = 4 if etype == 4 else 10
            conn = nodes.reshape(-1, nn)[:, :4]
            return np.vectorize(tag_to_idx.get)(conn)
    return None

vol_tets = {}
for vt in vol_tags:
    tets = get_tets_from_volume(vt)
    if tets is not None and len(tets) > 0:
        vol_tets[vt] = tets

if len(vol_tets) < 2:
    safe_finalize()
    raise RuntimeError("Could not extract tetrahedra from >=2 volumes.")

# Classify volumes as paste vs solid by mean radius
def volume_mean_radius(tets):
    pts = P_all[tets].mean(axis=1)
    r = np.sqrt((pts[:, 0] - xc) ** 2 + pts[:, 1] ** 2)
    return float(r.mean())

vol_info = [(vt, volume_mean_radius(tets), tets.shape[0]) for vt, tets in vol_tets.items()]
vol_info.sort(key=lambda x: x[1])

paste_vol_tag = vol_info[0][0]
solid_vol_tags = [v[0] for v in vol_info[1:]]

tets_paste_all = vol_tets[paste_vol_tag]
tets_solid_all = np.vstack([vol_tets[vt] for vt in solid_vol_tags])

print(f"Mesh total nodes (gmsh): {P_all.shape[0]}")
print(f"Tetrahedra: solid={len(tets_solid_all)}, paste={len(tets_paste_all)}")

# Extract boundary triangles (dim=2)
elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
tri_faces_all = []
for etype, nodes in zip(elem_types, elem_node_tags):
    if etype in (2, 9):
        nn = 3 if etype == 2 else 6
        conn = nodes.reshape(-1, nn)[:, :3]
        tri_faces_all.append(np.vectorize(tag_to_idx.get)(conn))
tri_faces_all = np.vstack(tri_faces_all) if tri_faces_all else np.empty((0, 3), dtype=int)

safe_finalize()

# -------------------- ACTIVE NODE REDUCTION ---------------------
used = set(tets_solid_all.ravel().tolist()) | set(tets_paste_all.ravel().tolist()) | set(tri_faces_all.ravel().tolist())
used = np.array(sorted(used), dtype=int)

new_index = -np.ones(P_all.shape[0], dtype=int)
new_index[used] = np.arange(len(used))

P = P_all[used]
tets_solid = new_index[tets_solid_all]
tets_paste = new_index[tets_paste_all]
tri_faces  = new_index[tri_faces_all]

Nn = P.shape[0]
print(f"Active nodes used in FEM: {Nn}")

# -------------------- CLASSIFY BOUNDARY TRIANGLES ---------------------
tol = 5e-10
Pc = P[tri_faces].mean(axis=1)
rc = np.sqrt((Pc[:, 0] - xc) ** 2 + Pc[:, 1] ** 2)

is_sym   = np.isclose(Pc[:, 0], xc, atol=tol)
is_inner = np.isclose(rc, R_in, atol=5e-6) & (Pc[:, 1] >= -tol)
is_outer = ~(is_sym | is_inner)

tri_inner = tri_faces[is_inner]   # Neumann
tri_outer = tri_faces[is_outer]   # Robin

print("Boundary triangles classified:")
print(f"  inner (flux) : {len(tri_inner)}")
print(f"  outer (conv) : {len(tri_outer)}")
print(f"  symmetry     : {int(is_sym.sum())} (natural Neumann)")


# -------------------- MIRROR HELPERS ---------------------
def mirror_full(P_half, xc_val):
    on_sym = np.isclose(P_half[:, 0], xc_val, atol=1e-12)
    idx_map = np.full(P_half.shape[0], -1, dtype=int)

    P_full = P_half.copy()
    for i in range(P_half.shape[0]):
        if on_sym[i]:
            idx_map[i] = i
        else:
            idx_map[i] = P_full.shape[0]
            P_full = np.vstack([P_full, [2.0 * xc_val - P_half[i, 0], P_half[i, 1], P_half[i, 2]]])
    return P_full, idx_map


# -------------------- PYVISTA PREVIEW (1) ---------------------
try:
    import pyvista as pv
    HAVE_PV = True
except Exception:
    HAVE_PV = False

def rot_matrix_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

def rot_matrix_x(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)

if HAVE_PV:
    P_full_preview, idx_map_preview = mirror_full(P, xc)
    Pmm_preview = P_full_preview * 1e3

    tets_half_all_preview = np.vstack([tets_solid, tets_paste])
    tets_m_preview = np.column_stack([
        idx_map_preview[tets_half_all_preview[:, 0]],
        idx_map_preview[tets_half_all_preview[:, 2]],
        idx_map_preview[tets_half_all_preview[:, 1]],
        idx_map_preview[tets_half_all_preview[:, 3]],
    ])
    tets_full_preview = np.vstack([tets_half_all_preview, tets_m_preview])

    cells_preview = np.hstack([np.full((tets_full_preview.shape[0], 1), 4, dtype=np.int64), tets_full_preview]).ravel()
    celltypes_preview = np.full(tets_full_preview.shape[0], pv.CellType.TETRA, dtype=np.uint8)

    grid_preview = pv.UnstructuredGrid(cells_preview, celltypes_preview, Pmm_preview)

    if apply_rotation_preview:
        Rz = rot_matrix_z(rot_xy_deg)
        Rx = rot_matrix_x(rot_yz_deg)
        R  = Rx @ Rz
        grid_preview.points = (grid_preview.points @ R.T)

    surf_preview = grid_preview.extract_surface().triangulate()

    pprev = pv.Plotter()
    pprev.add_text("1) FULL mesh preview (close -> solve transient + temperature)", font_size=10)
    pprev.add_mesh(surf_preview, show_edges=True, color="white", opacity=1.0)
    pprev.view_isometric()
    pprev.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
    pprev.show()


# -------------------- ASSEMBLE: K, M, and time-independent parts ---------------------
KI, KJ, KV = [], [], []
MI, MJ, MV = [], [], []

# f_conv is constant in time (h*Tinf term); f_flux will be assembled per time step from q_in(t)
f_conv = np.zeros(Nn, dtype=float)

def tet_volume(X4):
    A = np.array([
        [1.0, X4[0, 0], X4[0, 1], X4[0, 2]],
        [1.0, X4[1, 0], X4[1, 1], X4[1, 2]],
        [1.0, X4[2, 0], X4[2, 1], X4[2, 2]],
        [1.0, X4[3, 0], X4[3, 1], X4[3, 2]],
    ], dtype=float)
    return abs(np.linalg.det(A)) / 6.0

# Consistent mass matrix for linear tet:
Mloc = np.array([[2, 1, 1, 1],
                 [1, 2, 1, 1],
                 [1, 1, 2, 1],
                 [1, 1, 1, 2]], dtype=float)

def add_tet_conduction_and_mass(tets, k_mat, rc_mat):
    for tet in tets:
        i1, i2, i3, i4 = tet
        X = P[[i1, i2, i3, i4], :]

        # volume
        V = tet_volume(X)
        if V <= 0:
            continue

        # gradients for stiffness
        A = np.array([
            [1.0, X[0, 0], X[0, 1], X[0, 2]],
            [1.0, X[1, 0], X[1, 1], X[1, 2]],
            [1.0, X[2, 0], X[2, 1], X[2, 2]],
            [1.0, X[3, 0], X[3, 1], X[3, 2]],
        ], dtype=float)
        invA = np.linalg.inv(A)
        grads = invA[1:, :].T  # 4x3

        Ke = k_mat * V * (grads @ grads.T)
        Me = rc_mat * (V / 20.0) * Mloc

        nodes = [i1, i2, i3, i4]
        for a in range(4):
            for b in range(4):
                KI.append(nodes[a]); KJ.append(nodes[b]); KV.append(Ke[a, b])
                MI.append(nodes[a]); MJ.append(nodes[b]); MV.append(Me[a, b])

def tri_area(tri):
    a, b, c = tri
    pa, pb, pc_ = P[a], P[b], P[c]
    return 0.5 * np.linalg.norm(np.cross(pb - pa, pc_ - pa))

# Assemble conduction + mass
add_tet_conduction_and_mass(tets_solid, k_s, rc_s)
add_tet_conduction_and_mass(tets_paste, k_p, rc_p)

# Robin on OUTER adds to K and to f_conv
Hloc = np.array([[2, 1, 1],
                 [1, 2, 1],
                 [1, 1, 2]], dtype=float)

for tri in tri_outer:
    a, b, c = tri
    Atri = float(tri_area(tri))

    Kh = (h * Atri / 12.0) * Hloc
    fh = (h * Tinf * Atri / 3.0) * np.array([1.0, 1.0, 1.0], dtype=float)

    f_conv[a] += fh[0]; f_conv[b] += fh[1]; f_conv[c] += fh[2]

    nodes = [a, b, c]
    for i in range(3):
        for j in range(3):
            KI.append(nodes[i]); KJ.append(nodes[j]); KV.append(Kh[i, j])

K = coo_matrix((KV, (KI, KJ)), shape=(Nn, Nn)).tocsr()
M = coo_matrix((MV, (MI, MJ)), shape=(Nn, Nn)).tocsr()

# -------------------- Time-dependent Neumann flux assembly helper ---------------------
# For inner flux, the spatial integral pattern is the same; only scalar q_in(t) changes.
# Precompute areas and node triplets for speed.
inner_tris = tri_inner.copy()
inner_areas = np.array([tri_area(tri) for tri in inner_tris], dtype=float)

def q_in_time(t):
    """Flux ramp: 0 -> q_in0 over ramp_time (seconds)."""
    if ramp_time is None or ramp_time <= 0.0:
        return float(q_in0)
    r = min(max(t / ramp_time, 0.0), 1.0)
    return float(q_in0) * r

def build_f_flux(qval):
    f_flux = np.zeros(Nn, dtype=float)
    qval = float(qval)
    for tri, Atri in zip(inner_tris, inner_areas):
        a, b, c = tri
        fq = (qval * float(Atri) / 3.0)
        f_flux[a] += fq; f_flux[b] += fq; f_flux[c] += fq
    return f_flux

# -------------------- TRANSIENT SOLVE ---------------------
Nt = int(np.ceil(t_end / dt))
print(f"Time steps: Nt={Nt} (dt={dt}, t_end={t_end})")

Tn = np.full(Nn, float(T0), dtype=float)

# Build time-stepping matrices
if time_scheme.upper() == "BE":
    A = (M / dt + K).tocsr()
    # solve loop
    for n in range(Nt):
        t_np1 = (n + 1) * dt
        f_np1 = f_conv + build_f_flux(q_in_time(t_np1))
        rhs = f_np1 + (M / dt).dot(Tn)
        Tn = spsolve(A, rhs)

elif time_scheme.upper() == "CN":
    A = (M / dt + 0.5 * K).tocsr()
    B = (M / dt - 0.5 * K).tocsr()
    for n in range(Nt):
        t_n = n * dt
        t_np1 = (n + 1) * dt
        f_n   = f_conv + build_f_flux(q_in_time(t_n))
        f_np1 = f_conv + build_f_flux(q_in_time(t_np1))
        rhs = B.dot(Tn) + 0.5 * (f_n + f_np1)
        Tn = spsolve(A, rhs)
else:
    raise ValueError("time_scheme must be 'BE' or 'CN'.")

T = Tn
print(f"[FINAL @ t={Nt*dt:.3g}s] Tmin={T.min():.8f} °C, Tmax={T.max():.8f} °C, span={(T.max()-T.min()):.3e} °C")


# -------------------- BUILD FULL MIRROR FOR OUTPUT + VIS ---------------------
P_full, idx_map = mirror_full(P, xc)

T_full = np.zeros(P_full.shape[0], dtype=float)
T_full[:Nn] = T
for i in range(Nn):
    if idx_map[i] != i:
        T_full[idx_map[i]] = T[i]

tets_half_all = np.vstack([tets_solid, tets_paste])
tets_m = np.column_stack([
    idx_map[tets_half_all[:, 0]],
    idx_map[tets_half_all[:, 2]],
    idx_map[tets_half_all[:, 1]],
    idx_map[tets_half_all[:, 3]],
])
tets_full = np.vstack([tets_half_all, tets_m])


# -------------------- SAVE VTU (FULL, meters) ---------------------
def write_vtu_unstructured(points, tets, scalars, scalar_name, filename):
    Np = points.shape[0]
    Nc = tets.shape[0]
    connectivity = tets.ravel()
    offsets = np.arange(1, Nc + 1, dtype=int) * 4
    types = np.full(Nc, 10, dtype=int)  # VTK_TETRA=10

    with open(filename, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{Np}" NumberOfCells="{Nc}">\n')

        f.write(f'      <PointData Scalars="{scalar_name}">\n')
        f.write(f'        <DataArray type="Float64" Name="{scalar_name}" format="ascii">\n')
        f.write("          " + " ".join(f"{v:.12e}" for v in scalars) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </PointData>\n')

        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for p in points:
            f.write(f"          {p[0]:.12e} {p[1]:.12e} {p[2]:.12e}\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write("          " + " ".join(str(int(v)) for v in connectivity) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write("          " + " ".join(str(int(v)) for v in offsets) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + " ".join(str(int(v)) for v in types) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')

        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

write_vtu_unstructured(P_full, tets_full, T_full, "T", save_vtu_path)
print(f"Saved VTU: {save_vtu_path}")


# -------------------- PYVISTA GUI ---------------------
if not HAVE_PV:
    print("PyVista not available. Open the VTU in ParaView.")
    raise SystemExit

# Convert to mm for display
Pmm = P_full * 1e3

if apply_rotation_gui:
    Rz = rot_matrix_z(rot_xy_deg)
    Rx = rot_matrix_x(rot_yz_deg)
    R  = Rx @ Rz
    Pmm = (Pmm @ R.T)

cells = np.hstack([np.full((tets_full.shape[0], 1), 4, dtype=np.int64), tets_full]).ravel()
celltypes = np.full(tets_full.shape[0], pv.CellType.TETRA, dtype=np.uint8)

grid = pv.UnstructuredGrid(cells, celltypes, Pmm)
grid.point_data["T"]  = T_full.astype(float)
grid.point_data["dT"] = (T_full - Tinf).astype(float)

print("grid arrays:", grid.array_names)
print(f"[GRID] T: min={grid.point_data['T'].min():.7g}, max={grid.point_data['T'].max():.7g}")
print(f"[GRID] dT: min={grid.point_data['dT'].min():.7g}, max={grid.point_data['dT'].max():.7g}")

def robust_clim(vals, use_percentile=True, plo=1.0, phi=99.0, min_span=1e-6, pad_frac=0.02):
    vals = np.asarray(vals, dtype=float)
    if use_percentile:
        lo, hi = np.percentile(vals, [plo, phi])
    else:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    lo = float(lo); hi = float(hi)
    span = hi - lo
    if not np.isfinite(span) or span < min_span:
        mid = 0.5 * (lo + hi)
        lo = mid - 0.5 * min_span
        hi = mid + 0.5 * min_span
        span = hi - lo
    pad = pad_frac * span
    return (lo - pad, hi + pad)

# -------------------- (2) Whole-object temperature preview ---------------------
geom = grid.extract_geometry()
surf_all = geom.extract_surface().clean()

T_surf = np.asarray(surf_all.point_data["T"], float)
T_CLIM_SURF = robust_clim(T_surf, USE_PERCENTILE_CLIM, PCT_LO, PCT_HI, MIN_SPAN, PAD_FRAC)

pT = pv.Plotter()
pT.add_text(f"2) Whole object — Temperature T @ t={Nt*dt:.3g}s (close -> slicer view)", font_size=10)
pT.add_mesh(
    surf_all,
    scalars="T",
    preference="point",
    cmap=cmap_name,
    clim=T_CLIM_SURF,
    show_edges=False,
    opacity=1.0,
    scalar_bar_args={"title": "T (°C)", "fmt": SBAR_FMT},
)
pT.view_isometric()
pT.camera.zoom(1.2)
pT.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
pT.show()


# -------------------- (3) Interactive slicer view ---------------------
xmin, xmax = float(grid.points[:, 0].min()), float(grid.points[:, 0].max())
ymin, ymax = float(grid.points[:, 1].min()), float(grid.points[:, 1].max())
zmin, zmax = float(grid.points[:, 2].min()), float(grid.points[:, 2].max())

x0 = 0.5 * (xmin + xmax)
y0 = 0.5 * (ymin + ymax)
z0 = 0.5 * (zmin + zmax)

cut_state = {"x": x0, "y": y0, "z": z0}
invert_flag = (keep_side.lower() != "ge")

p = pv.Plotter()
p.add_text(f"3) Slicer view — Body:T + Slices:ΔT  (t={Nt*dt:.3g}s)", font_size=10)

body_actor = None
slice_x_actor = None
slice_y_actor = None
slice_z_actor = None

def safe_add_mesh(mesh, **kwargs):
    if mesh is None or getattr(mesh, "n_points", 0) == 0:
        return None
    return p.add_mesh(mesh, **kwargs)

def build_cut_xyz(x_mm, y_mm, z_mm):
    vol = grid
    vol = vol.clip(normal=(1, 0, 0), origin=(float(x_mm), 0.0, 0.0), invert=invert_flag)
    vol = vol.clip(normal=(0, 1, 0), origin=(0.0, float(y_mm), 0.0), invert=invert_flag)
    vol = vol.clip(normal=(0, 0, 1), origin=(0.0, 0.0, float(z_mm)), invert=invert_flag)

    body = vol.extract_geometry().extract_surface().clean()

    slx = grid.slice(normal=(1, 0, 0), origin=(float(x_mm), 0.0, 0.0))
    sly = grid.slice(normal=(0, 1, 0), origin=(0.0, float(y_mm), 0.0))
    slz = grid.slice(normal=(0, 0, 1), origin=(0.0, 0.0, float(z_mm)))

    return body, slx, sly, slz

def redraw():
    global body_actor, slice_x_actor, slice_y_actor, slice_z_actor

    x_mm = cut_state["x"]
    y_mm = cut_state["y"]
    z_mm = cut_state["z"]

    body, slx, sly, slz = build_cut_xyz(x_mm, y_mm, z_mm)

    if body_actor is not None:
        p.remove_actor(body_actor)
    if slice_x_actor is not None:
        p.remove_actor(slice_x_actor)
    if slice_y_actor is not None:
        p.remove_actor(slice_y_actor)
    if slice_z_actor is not None:
        p.remove_actor(slice_z_actor)

    if body is not None and body.n_points > 0:
        Tb = np.asarray(body.point_data["T"], float)
        T_CLIM_BODY = robust_clim(Tb, USE_PERCENTILE_CLIM, PCT_LO, PCT_HI, MIN_SPAN, PAD_FRAC)
    else:
        T_CLIM_BODY = T_CLIM_SURF

    if slx is not None and slx.n_points > 0:
        dTx = np.asarray(slx.point_data["dT"], float)
        dT_CLIM = robust_clim(dTx, USE_PERCENTILE_CLIM, PCT_LO, PCT_HI, MIN_SPAN, PAD_FRAC)
    else:
        dT_CLIM = robust_clim(grid.point_data["dT"], USE_PERCENTILE_CLIM, PCT_LO, PCT_HI, MIN_SPAN, PAD_FRAC)

    body_actor = safe_add_mesh(
        body,
        scalars="T",
        preference="point",
        cmap=cmap_name,
        clim=T_CLIM_BODY,
        show_edges=False,
        opacity=1.0,
        name="body",
        scalar_bar_args={"title": "T (°C)", "fmt": SBAR_FMT},
    )

    slice_x_actor = safe_add_mesh(
        slx,
        scalars="dT",
        preference="point",
        cmap=cmap_name,
        clim=dT_CLIM,
        show_edges=False,
        opacity=1.0,
        name="slice_x",
        reset_camera=False,
        show_scalar_bar=True,
        scalar_bar_args={"title": "ΔT (°C)", "fmt": SBAR_FMT},
    )
    slice_y_actor = safe_add_mesh(
        sly,
        scalars="dT",
        preference="point",
        cmap=cmap_name,
        clim=dT_CLIM,
        show_edges=False,
        opacity=1.0,
        name="slice_y",
        reset_camera=False,
        show_scalar_bar=False,
    )
    slice_z_actor = safe_add_mesh(
        slz,
        scalars="dT",
        preference="point",
        cmap=cmap_name,
        clim=dT_CLIM,
        show_edges=False,
        opacity=1.0,
        name="slice_z",
        reset_camera=False,
        show_scalar_bar=False,
    )

redraw()

def slider_x(val):
    cut_state["x"] = float(val)
    redraw()

def slider_y(val):
    cut_state["y"] = float(val)
    redraw()

def slider_z(val):
    cut_state["z"] = float(val)
    redraw()

p.add_slider_widget(
    slider_x, rng=[xmin, xmax], value=x0,
    title="Cut plane X (mm)", pointa=(0.03, 0.10), pointb=(0.35, 0.10),
    style="modern"
)
p.add_slider_widget(
    slider_y, rng=[ymin, ymax], value=y0,
    title="Cut plane Y (mm)", pointa=(0.03, 0.06), pointb=(0.35, 0.06),
    style="modern"
)
p.add_slider_widget(
    slider_z, rng=[zmin, zmax], value=z0,
    title="Cut plane Z (mm)", pointa=(0.03, 0.02), pointb=(0.35, 0.02),
    style="modern"
)

p.view_isometric()
p.camera.zoom(1.2)
p.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
p.show()
