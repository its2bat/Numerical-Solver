# ================================================================
# FEM Heat Transfer - BVP Project
# STEADY 3D FEM - 3-MATERIAL MODEL - FULL TUBE (Dirichlet inner wall)
#
# 2D cross-section (same as 2D solver) extruded along Z.
#
#   UPPER (y>0):  Mold(60Sn40Pb,k=50) > Paste(k=9) > Tube(SS316,k=16) > Water
#   LOWER (y<0):  Tube(SS316,k=16) exposed to air
#
# BCs:
#   INNER CYLINDER (R_in):       T = T_wall           [Dirichlet]
#   MOLD OUTER + Z-ENDS:         -k dT/dn = h(T-Tinf) [Robin]
#   PASTE BOTTOM (y=0):          -k dT/dn = h(T-Tinf) [Robin]
#   EXPOSED TUBE ARC (y<0):      -k dT/dn = h(T-Tinf) [Robin]
#   SYMMETRY (x=xc):             dT/dn = 0            [natural]
#
# Outputs:
#   VTU file  +  PyVista interactive visualization
# ================================================================

import numpy as np
import gmsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os

# ======================== PARAMETERS ========================
# Geometry (mm)
W_mm  = 1.4
H_mm  = 1.0
Lz_mm = 1.0

tube_OD_mm = 0.8
tube_ID_mm = 0.5
paste_thickness_mm = 0.1

R_in_mm    = tube_ID_mm / 2.0
R_tube_mm  = tube_OD_mm / 2.0
R_paste_mm = R_tube_mm + paste_thickness_mm

# Materials
k_tube   = 16.0    # W/mK  (SS316)
k_paste  = 9.0     # W/mK
k_solder = 50.0    # W/mK  (60Sn/40Pb)

# Convection (air)
h_conv = 15.0      # W/m2K
Tinf   = 25.0      # C

# Dirichlet inner wall
T_wall = 60.0      # C

# Mesh
h_global = 0.06e-3
h_arc    = 0.008e-3
h_mid    = 0.015e-3
dist_min = 0.02e-3
dist_max = 0.15e-3
nz_layers = 28

# Output
save_dir = os.path.dirname(os.path.abspath(__file__))
save_vtu = os.path.join(save_dir, "Steady3D_Twall_3mat_full.vtu")
cmap_name = "turbo"

# ======================== UNIT CONVERSION ========================
W  = W_mm * 1e-3;  H = H_mm * 1e-3;  Lz = Lz_mm * 1e-3
R_in    = R_in_mm    * 1e-3
R_tube  = R_tube_mm  * 1e-3
R_paste = R_paste_mm * 1e-3
xc = W / 2.0

print("=" * 60)
print("STEADY 3D - 3-material, full tube, Dirichlet inner wall")
print("=" * 60)
print(f"Domain: {W_mm}x{H_mm}x{Lz_mm} mm")
print(f"Tube: ID={tube_ID_mm}, OD={tube_OD_mm}, paste={paste_thickness_mm} mm")
print(f"k: tube={k_tube}, paste={k_paste}, solder={k_solder}")
print(f"BC: T_wall={T_wall}C, h={h_conv}, Tinf={Tinf}C")

# ================================================================
# 1) BUILD 2D CROSS-SECTION + EXTRUDE TO 3D
# ================================================================
try:
    gmsh.finalize()
except Exception:
    pass

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("steady_3d_3mat")

# ---- 2D Points (same as 2D solver) ----
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

# ---- 2D Surfaces ----
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

# ---- Extrude all 4 surfaces along Z ----
all_2d = [(2, s_tube_u), (2, s_tube_l), (2, s_paste), (2, s_mold)]
ext = gmsh.model.geo.extrude(all_2d, 0, 0, Lz, [nz_layers])

gmsh.model.geo.synchronize()

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

print("\nMeshing 3D ...")
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Laplace2D")

# ================================================================
# 2) EXTRACT MESH
# ================================================================
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
P_all = node_coords.reshape(-1, 3)
tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

# All tets (from all 3D entities)
all_tets = []
for dim, tag in gmsh.model.getEntities(3):
    etypes, _, enodes = gmsh.model.mesh.getElements(3, tag)
    for etype, nodes in zip(etypes, enodes):
        if etype in (4, 11):
            nn = 4 if etype == 4 else 10
            conn = nodes.reshape(-1, nn)[:, :4].astype(int)
            all_tets.append(np.vectorize(tag_to_idx.get)(conn))
all_tets = np.vstack(all_tets)

# All boundary triangles (from all 2D entities)
all_tris = []
for dim, tag in gmsh.model.getEntities(2):
    etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
    for etype, nodes in zip(etypes, enodes):
        if etype in (2, 9):
            nn = 3 if etype == 2 else 6
            conn = nodes.reshape(-1, nn)[:, :3].astype(int)
            all_tris.append(np.vectorize(tag_to_idx.get)(conn))
all_tris = np.vstack(all_tris) if all_tris else np.empty((0, 3), dtype=int)

gmsh.finalize()

# ================================================================
# 3) ACTIVE NODE REDUCTION
# ================================================================
used = set(all_tets.ravel()) | set(all_tris.ravel())
used = np.array(sorted(used), dtype=int)

new_idx = -np.ones(P_all.shape[0], dtype=int)
new_idx[used] = np.arange(len(used))

P        = P_all[used]
all_tets = new_idx[all_tets]
all_tris = new_idx[all_tris]
Nn = P.shape[0]

# ================================================================
# 4) CLASSIFY TETS BY MATERIAL (centroid radius from tube center)
# ================================================================
tet_centroids = P[all_tets].mean(axis=1)  # (Ntet, 3)
r_tet = np.sqrt((tet_centroids[:, 0] - xc)**2 + tet_centroids[:, 1]**2)

r_mid_tube_paste = 0.5 * (R_tube + R_paste)
r_mid_in_tube    = 0.5 * (R_in + R_tube)

mask_tube  = r_tet < r_mid_tube_paste   # tube: r < midpoint(tube,paste)
mask_paste = (r_tet >= r_mid_tube_paste) & (r_tet < R_paste + 0.5*(R_paste - R_tube))
# Refine: tube is between R_in and R_tube, paste between R_tube and R_paste
mask_tube  = r_tet < r_mid_tube_paste
mask_paste = (~mask_tube) & (r_tet < R_paste + 0.3 * (R_paste - R_tube))
mask_mold  = ~mask_tube & ~mask_paste

tets_tube  = all_tets[mask_tube]
tets_paste = all_tets[mask_paste]
tets_mold  = all_tets[mask_mold]

print(f"\nMesh: {Nn} nodes, {len(all_tets)} tets")
print(f"  tube: {len(tets_tube)}, paste: {len(tets_paste)}, mold: {len(tets_mold)}")

# ================================================================
# 5) CLASSIFY BOUNDARY TRIANGLES
# ================================================================
tri_c = P[all_tris].mean(axis=1)  # centroids
r_tri = np.sqrt((tri_c[:, 0] - xc)**2 + tri_c[:, 1]**2)

tol_r = 5e-6   # tolerance for radius matching
tol_c = 5e-10  # tolerance for coordinate matching

# Dirichlet: inner cylinder (r ~ R_in) at any z, any y
is_dirichlet = np.isclose(r_tri, R_in, atol=tol_r)

# Symmetry: x ~ xc
is_sym = np.isclose(tri_c[:, 0], xc, atol=tol_c) & ~is_dirichlet

# Z-end faces (z ~ 0 or z ~ Lz) — Robin (all except symmetry & inner)
is_z0 = np.isclose(tri_c[:, 2], 0.0, atol=tol_c)
is_zL = np.isclose(tri_c[:, 2], Lz, atol=tol_c)
is_z_end = (is_z0 | is_zL) & ~is_dirichlet & ~is_sym

# Lateral outer surfaces (0 < z < Lz):
not_special = ~is_dirichlet & ~is_sym & ~is_z_end
is_mold_right   = not_special & np.isclose(tri_c[:, 0], W, atol=tol_c)
is_mold_top     = not_special & np.isclose(tri_c[:, 1], H, atol=tol_c)
is_mold_bot     = not_special & np.isclose(tri_c[:, 1], 0.0, atol=tol_c) & (r_tri > R_paste - tol_r)
is_paste_bot    = not_special & np.isclose(tri_c[:, 1], 0.0, atol=tol_c) & \
                  (r_tri > R_tube - tol_r) & (r_tri < R_paste + tol_r) & ~is_mold_bot
is_tube_exposed = not_special & np.isclose(r_tri, R_tube, atol=tol_r) & (tri_c[:, 1] < -tol_c)

is_robin = is_z_end | is_mold_right | is_mold_top | is_mold_bot | is_paste_bot | is_tube_exposed

tri_dir = all_tris[is_dirichlet]
tri_rob = all_tris[is_robin]

n_internal = np.sum(~is_dirichlet & ~is_sym & ~is_robin)
print(f"\nBoundary triangles:")
print(f"  Dirichlet (inner): {len(tri_dir)}")
print(f"  Robin (outer+z):   {len(tri_rob)}")
print(f"  Symmetry:          {int(is_sym.sum())}")
print(f"  Internal (no BC):  {n_internal}")

# ================================================================
# 6) VECTORIZED FEM ASSEMBLY (3D linear tets)
# ================================================================
f = np.zeros(Nn)
K_rows, K_cols, K_vals = [], [], []

# Reference gradient matrix for linear tet: dN/d(xi,eta,zeta)
G_ref = np.array([[-1., -1., -1.],
                   [ 1.,  0.,  0.],
                   [ 0.,  1.,  0.],
                   [ 0.,  0.,  1.]])  # (4, 3)


def assemble_K_tet(tets, k_val):
    """Vectorized stiffness assembly for linear tetrahedra."""
    if len(tets) == 0:
        return np.array([]), np.array([]), np.array([])

    i1, i2, i3, i4 = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]

    # Jacobian: J[e] = [[x2-x1, x3-x1, x4-x1], [y.., ..], [z.., ..]]
    d1 = P[i2] - P[i1]  # (Ne, 3)
    d2 = P[i3] - P[i1]
    d3 = P[i4] - P[i1]
    J = np.stack([d1, d2, d3], axis=2)  # (Ne, 3, 3)

    detJ = np.linalg.det(J)  # (Ne,)
    V = np.abs(detJ) / 6.0
    valid = V > 0

    J_v   = J[valid]
    V_v   = V[valid]
    tets_v = tets[valid]

    # Inverse Jacobian
    J_inv = np.linalg.inv(J_v)  # (Nv, 3, 3)

    # Physical gradients: G_phys = G_ref @ J_inv  -> (Nv, 4, 3)
    G_phys = np.einsum('ij,ejk->eik', G_ref, J_inv)

    # Element stiffness: K_e = k * V * G_phys @ G_phys^T  -> (Nv, 4, 4)
    Ke = k_val * V_v[:, None, None] * np.einsum('eij,ekj->eik', G_phys, G_phys)

    # COO indices
    nd = tets_v  # (Nv, 4)
    rows = np.repeat(nd, 4, axis=1).ravel()
    cols = np.tile(nd, (1, 4)).ravel()

    return rows, cols, Ke.ravel()


for tets, k, name in [(tets_tube, k_tube, "tube"),
                       (tets_paste, k_paste, "paste"),
                       (tets_mold, k_solder, "mold")]:
    r, c, v = assemble_K_tet(tets, k)
    K_rows.append(r); K_cols.append(c); K_vals.append(v)
    print(f"  assembled {name}: {len(tets)} tets, k={k}")

# Robin BC on boundary triangles
H_template = np.array([[2., 1., 1.],
                        [1., 2., 1.],
                        [1., 1., 2.]])  # consistent boundary mass

for tri_set in [tri_rob]:
    if len(tri_set) == 0:
        continue
    a, b, c = tri_set[:, 0], tri_set[:, 1], tri_set[:, 2]
    pa, pb, pc_ = P[a], P[b], P[c]
    cross = np.cross(pb - pa, pc_ - pa)
    A_tri = 0.5 * np.sqrt(np.sum(cross**2, axis=1))  # triangle areas

    # Stiffness contribution: h * A/12 * [[2,1,1],[1,2,1],[1,1,2]]
    coeff = h_conv * A_tri / 12.0
    Kh = coeff[:, None, None] * H_template[None, :, :]  # (Nt, 3, 3)

    nd = tri_set  # (Nt, 3)
    rows = np.repeat(nd, 3, axis=1).ravel()
    cols = np.tile(nd, (1, 3)).ravel()
    K_rows.append(rows); K_cols.append(cols); K_vals.append(Kh.ravel())

    # Load: h * Tinf * A/3 per node
    fh = h_conv * Tinf * A_tri / 3.0
    np.add.at(f, a, fh)
    np.add.at(f, b, fh)
    np.add.at(f, c, fh)

# Build global K
K = coo_matrix(
    (np.concatenate(K_vals),
     (np.concatenate(K_rows).astype(int),
      np.concatenate(K_cols).astype(int))),
    shape=(Nn, Nn)
).tocsr()

# ================================================================
# 7) DIRICHLET BC + SOLVE
# ================================================================
dir_nodes = np.unique(tri_dir.ravel())
free_mask = np.ones(Nn, dtype=bool)
free_mask[dir_nodes] = False
free_idx = np.where(free_mask)[0]
Nf, Nd = len(free_idx), len(dir_nodes)

print(f"\nDirichlet nodes: {Nd},  Free nodes: {Nf}")
print("Solving ...")

K_ff = K[free_idx, :][:, free_idx]
rhs  = f[free_idx] - K[free_idx, :][:, dir_nodes] @ np.full(Nd, T_wall)

T = np.full(Nn, T_wall, dtype=float)
T[free_idx] = spsolve(K_ff, rhs)

print(f"\nSolution: T_min={T.min():.4f}C, T_max={T.max():.4f}C")

# Energy balance
q_in = np.sum(K[dir_nodes, :] @ T - f[dir_nodes])
print(f"Heat in (reaction): {q_in:.6e} W")

# ================================================================
# 8) MIRROR -> FULL DOMAIN (vectorized)
# ================================================================
on_sym  = np.isclose(P[:, 0], xc, atol=1e-12)
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

# Mirror tets (swap nodes 1,2 to flip orientation)
tets_m = np.column_stack([
    mirror_map[all_tets[:, 0]],
    mirror_map[all_tets[:, 2]],
    mirror_map[all_tets[:, 1]],
    mirror_map[all_tets[:, 3]],
])
tets_full = np.vstack([all_tets, tets_m])

# ================================================================
# 9) SAVE VTU
# ================================================================
def write_vtu(points, tets, scalars_dict, filename):
    Np, Nc = points.shape[0], tets.shape[0]
    with open(filename, "w", encoding="utf-8") as fout:
        fout.write('<?xml version="1.0"?>\n')
        fout.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        fout.write('  <UnstructuredGrid>\n')
        fout.write(f'    <Piece NumberOfPoints="{Np}" NumberOfCells="{Nc}">\n')
        fout.write('      <PointData>\n')
        for name, vals in scalars_dict.items():
            fout.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
            fout.write("          " + " ".join(f"{v:.8e}" for v in vals) + "\n")
            fout.write('        </DataArray>\n')
        fout.write('      </PointData>\n')
        fout.write('      <Points>\n')
        fout.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for p in points:
            fout.write(f"          {p[0]:.10e} {p[1]:.10e} {p[2]:.10e}\n")
        fout.write('        </DataArray>\n')
        fout.write('      </Points>\n')
        fout.write('      <Cells>\n')
        fout.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        fout.write("          " + " ".join(str(int(v)) for v in tets.ravel()) + "\n")
        fout.write('        </DataArray>\n')
        fout.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        fout.write("          " + " ".join(str(i * 4) for i in range(1, Nc + 1)) + "\n")
        fout.write('        </DataArray>\n')
        fout.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        fout.write("          " + " ".join("10" for _ in range(Nc)) + "\n")
        fout.write('        </DataArray>\n')
        fout.write('      </Cells>\n')
        fout.write('    </Piece>\n')
        fout.write('  </UnstructuredGrid>\n')
        fout.write('</VTKFile>\n')

write_vtu(P_full, tets_full, {"T": T_full, "dT": T_full - Tinf}, save_vtu)
print(f"VTU saved: {save_vtu}")

# ================================================================
# 10) PYVISTA VISUALIZATION
# ================================================================
try:
    import pyvista as pv
    HAVE_PV = True
except ImportError:
    HAVE_PV = False
    print("PyVista not installed. Open VTU in ParaView.")

if HAVE_PV:
    Pmm = P_full * 1e3  # convert to mm for display

    cells = np.hstack([
        np.full((tets_full.shape[0], 1), 4, dtype=np.int64),
        tets_full
    ]).ravel()
    celltypes = np.full(tets_full.shape[0], pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, Pmm)
    grid.point_data["T"]  = T_full
    grid.point_data["dT"] = T_full - Tinf

    def robust_clim(vals, plo=1.0, phi=99.0, min_span=1e-6):
        lo, hi = np.percentile(vals, [plo, phi])
        if hi - lo < min_span:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
        pad = 0.02 * (hi - lo)
        return (lo - pad, hi + pad)

    # --- 1) Mesh preview ---
    surf = grid.extract_surface().triangulate()
    p1 = pv.Plotter()
    p1.add_text("1) FULL mesh preview (close -> temperature view)", font_size=10)
    p1.add_mesh(surf, show_edges=True, color="white", opacity=1.0)
    p1.view_isometric()
    p1.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
    p1.show()

    # --- 2) Temperature preview ---
    surf_all = grid.extract_surface(algorithm=None).clean()
    T_surf = np.asarray(surf_all.point_data["T"], float)
    T_CLIM = robust_clim(T_surf)

    p2 = pv.Plotter()
    p2.add_text("2) Temperature T (close -> interactive slicer)", font_size=10)
    p2.add_mesh(surf_all, scalars="T", preference="point",
                cmap=cmap_name, clim=T_CLIM, show_edges=False,
                scalar_bar_args={"title": "T (C)", "fmt": "%.3g"})
    p2.view_isometric()
    p2.camera.zoom(1.2)
    p2.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
    p2.show()

    # --- 3) Interactive slicer ---
    xmin, xmax = float(Pmm[:, 0].min()), float(Pmm[:, 0].max())
    ymin, ymax = float(Pmm[:, 1].min()), float(Pmm[:, 1].max())
    zmin, zmax = float(Pmm[:, 2].min()), float(Pmm[:, 2].max())
    x0, y0, z0 = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)

    cut_state = {"x": x0, "y": y0, "z": z0}
    p3 = pv.Plotter()
    p3.add_text("3) Interactive slicer: Body=T, Slices=dT", font_size=10)

    body_actor = [None]
    slice_actors = [None, None, None]

    def redraw():
        x_mm, y_mm, z_mm = cut_state["x"], cut_state["y"], cut_state["z"]

        # Clip body
        vol = grid
        vol = vol.clip(normal=(1, 0, 0), origin=(x_mm, 0, 0), invert=False)
        vol = vol.clip(normal=(0, 1, 0), origin=(0, y_mm, 0), invert=False)
        vol = vol.clip(normal=(0, 0, 1), origin=(0, 0, z_mm), invert=False)
        body = vol.extract_surface(algorithm=None).clean()

        # Slices
        slx = grid.slice(normal=(1, 0, 0), origin=(x_mm, 0, 0))
        sly = grid.slice(normal=(0, 1, 0), origin=(0, y_mm, 0))
        slz = grid.slice(normal=(0, 0, 1), origin=(0, 0, z_mm))

        # Remove old actors
        if body_actor[0] is not None:
            p3.remove_actor(body_actor[0])
        for i in range(3):
            if slice_actors[i] is not None:
                p3.remove_actor(slice_actors[i])

        # Compute clims from actual displayed data
        if body is not None and body.n_points > 0:
            T_clim = robust_clim(np.asarray(body.point_data["T"], float))
        else:
            T_clim = T_CLIM

        if slx is not None and slx.n_points > 0:
            dT_clim = robust_clim(np.asarray(slx.point_data["dT"], float))
        else:
            dT_clim = robust_clim(grid.point_data["dT"])

        # Add body
        if body is not None and body.n_points > 0:
            body_actor[0] = p3.add_mesh(
                body, scalars="T", preference="point", cmap=cmap_name,
                clim=T_clim, show_edges=False, name="body",
                scalar_bar_args={"title": "T (C)", "fmt": "%.3g"})

        # Add slices
        for i, (sl, show_bar) in enumerate([(slx, True), (sly, False), (slz, False)]):
            if sl is not None and sl.n_points > 0:
                slice_actors[i] = p3.add_mesh(
                    sl, scalars="dT", preference="point", cmap=cmap_name,
                    clim=dT_clim, show_edges=False, name=f"slice_{i}",
                    reset_camera=False, show_scalar_bar=show_bar,
                    scalar_bar_args={"title": "dT (C)", "fmt": "%.3g"})

    redraw()

    def slider_x(val): cut_state["x"] = float(val); redraw()
    def slider_y(val): cut_state["y"] = float(val); redraw()
    def slider_z(val): cut_state["z"] = float(val); redraw()

    p3.add_slider_widget(slider_x, rng=[xmin, xmax], value=x0,
                         title="X (mm)", pointa=(0.03, 0.10), pointb=(0.35, 0.10))
    p3.add_slider_widget(slider_y, rng=[ymin, ymax], value=y0,
                         title="Y (mm)", pointa=(0.03, 0.06), pointb=(0.35, 0.06))
    p3.add_slider_widget(slider_z, rng=[zmin, zmax], value=z0,
                         title="Z (mm)", pointa=(0.03, 0.02), pointb=(0.35, 0.02))

    p3.view_isometric()
    p3.camera.zoom(1.2)
    p3.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
    p3.show()

print("\nDone.")
