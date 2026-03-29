# ================================================================
# Math 543 - BVP Project
# TRANSIENT 3D FEM - 3-MATERIAL MODEL - FULL TUBE (Dirichlet inner wall)
#
# Same geometry as steady 3D: 2D cross-section extruded along Z.
# Backward Euler with convergence auto-stop.
#
# PDE:  rho*cp * dT/dt = div(k grad T)
#
# BCs:
#   INNER CYLINDER (R_in):       T = T_wall           [Dirichlet]
#   MOLD OUTER + Z-ENDS:         -k dT/dn = h(T-Tinf) [Robin]
#   PASTE BOTTOM (y=0):          -k dT/dn = h(T-Tinf) [Robin]
#   EXPOSED TUBE ARC (y<0):      -k dT/dn = h(T-Tinf) [Robin]
#   SYMMETRY (x=xc):             dT/dn = 0            [natural]
#
# IC:  T(x,y,z,0) = T0
#
# Outputs:
#   VTU snapshots + convergence plot + PyVista visualization
# ================================================================

import numpy as np
import gmsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized
import os
import time as timer

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

# Materials (k, rho, cp)
k_tube   = 16.0;   rho_tube   = 8000.0;  cp_tube   = 500.0   # SS316
k_paste  = 9.0;    rho_paste  = 2500.0;  cp_paste  = 800.0   # thermal paste
k_solder = 50.0;   rho_solder = 8500.0;  cp_solder = 180.0   # 60Sn/40Pb

# Convection (air)
h_conv = 15.0
Tinf   = 25.0

# Dirichlet + IC
T_wall = 60.0
T0     = 25.0

# Time stepping
dt    = 0.05
t_end = 500.0
conv_tol = 1e-3   # auto-stop criterion (max |dT| per step)

# Mesh
h_global = 0.06e-3
h_arc    = 0.008e-3
h_mid    = 0.015e-3
dist_min = 0.02e-3
dist_max = 0.15e-3
nz_layers = 28

# Output
save_dir = os.path.dirname(os.path.abspath(__file__))
cmap_name = "turbo"

# ======================== UNIT CONVERSION ========================
W  = W_mm * 1e-3;  H = H_mm * 1e-3;  Lz = Lz_mm * 1e-3
R_in    = R_in_mm    * 1e-3
R_tube  = R_tube_mm  * 1e-3
R_paste = R_paste_mm * 1e-3
xc = W / 2.0

print("=" * 60)
print("TRANSIENT 3D - 3-material, full tube, Dirichlet inner wall")
print("=" * 60)
print(f"Domain: {W_mm}x{H_mm}x{Lz_mm} mm")
print(f"Tube: ID={tube_ID_mm}, OD={tube_OD_mm}, paste={paste_thickness_mm} mm")
print(f"k: tube={k_tube}, paste={k_paste}, solder={k_solder}")
print(f"BC: T_wall={T_wall}C, h={h_conv}, Tinf={Tinf}C, T0={T0}C")
print(f"Time: dt={dt}s, t_end={t_end}s, conv_tol={conv_tol}")

# ================================================================
# 1) BUILD 2D + EXTRUDE TO 3D  (identical geometry to steady)
# ================================================================
try:
    gmsh.finalize()
except Exception:
    pass

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("transient_3d_3mat")

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

arc_in_u   = gmsh.model.geo.addCircleArc(p_in_top,    pc, p_in_right)
arc_in_l   = gmsh.model.geo.addCircleArc(p_in_right,  pc, p_in_bot)
arc_tube_u = gmsh.model.geo.addCircleArc(p_tube_top,  pc, p_tube_right)
arc_tube_l = gmsh.model.geo.addCircleArc(p_tube_right, pc, p_tube_bot)
arc_paste  = gmsh.model.geo.addCircleArc(p_paste_top, pc, p_paste_right)

l_y0_tube  = gmsh.model.geo.addLine(p_in_right,    p_tube_right)
l_y0_paste = gmsh.model.geo.addLine(p_tube_right,  p_paste_right)
l_y0_mold  = gmsh.model.geo.addLine(p_paste_right, p_mold_br)
l_right = gmsh.model.geo.addLine(p_mold_br, p_mold_tr)
l_top   = gmsh.model.geo.addLine(p_mold_tr, p_mold_tl)
l_sym_mold   = gmsh.model.geo.addLine(p_mold_tl,  p_paste_top)
l_sym_paste  = gmsh.model.geo.addLine(p_paste_top, p_tube_top)
l_sym_tube_u = gmsh.model.geo.addLine(p_tube_top,  p_in_top)
l_sym_tube_l = gmsh.model.geo.addLine(p_tube_bot,  p_in_bot)

cl_tube_u = gmsh.model.geo.addCurveLoop([arc_in_u, l_y0_tube, -arc_tube_u, l_sym_tube_u])
s_tube_u = gmsh.model.geo.addPlaneSurface([cl_tube_u])
cl_tube_l = gmsh.model.geo.addCurveLoop([arc_tube_l, l_sym_tube_l, -arc_in_l, l_y0_tube])
s_tube_l = gmsh.model.geo.addPlaneSurface([cl_tube_l])
cl_paste = gmsh.model.geo.addCurveLoop([arc_tube_u, l_y0_paste, -arc_paste, l_sym_paste])
s_paste = gmsh.model.geo.addPlaneSurface([cl_paste])
cl_mold = gmsh.model.geo.addCurveLoop([arc_paste, l_y0_mold, l_right, l_top, l_sym_mold])
s_mold = gmsh.model.geo.addPlaneSurface([cl_mold])

gmsh.model.geo.synchronize()

ext = gmsh.model.geo.extrude(
    [(2, s_tube_u), (2, s_tube_l), (2, s_paste), (2, s_mold)],
    0, 0, Lz, [nz_layers])
gmsh.model.geo.synchronize()

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

all_tets = []
for dim, tag in gmsh.model.getEntities(3):
    etypes, _, enodes = gmsh.model.mesh.getElements(3, tag)
    for etype, nodes in zip(etypes, enodes):
        if etype in (4, 11):
            nn = 4 if etype == 4 else 10
            conn = nodes.reshape(-1, nn)[:, :4].astype(int)
            all_tets.append(np.vectorize(tag_to_idx.get)(conn))
all_tets = np.vstack(all_tets)

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
# 4) CLASSIFY TETS + BOUNDARY TRIS
# ================================================================
tet_c = P[all_tets].mean(axis=1)
r_tet = np.sqrt((tet_c[:, 0] - xc)**2 + tet_c[:, 1]**2)
r_mid = 0.5 * (R_tube + R_paste)
mask_tube  = r_tet < r_mid
mask_paste = (~mask_tube) & (r_tet < R_paste + 0.3*(R_paste - R_tube))
mask_mold  = ~mask_tube & ~mask_paste
tets_tube  = all_tets[mask_tube]
tets_paste = all_tets[mask_paste]
tets_mold  = all_tets[mask_mold]

tri_c = P[all_tris].mean(axis=1)
r_tri = np.sqrt((tri_c[:, 0] - xc)**2 + tri_c[:, 1]**2)
tol_r, tol_c = 5e-6, 5e-10

is_dirichlet = np.isclose(r_tri, R_in, atol=tol_r)
is_sym = np.isclose(tri_c[:, 0], xc, atol=tol_c) & ~is_dirichlet
is_z_end = (np.isclose(tri_c[:, 2], 0, atol=tol_c) | np.isclose(tri_c[:, 2], Lz, atol=tol_c)) & ~is_dirichlet & ~is_sym

not_special = ~is_dirichlet & ~is_sym & ~is_z_end
is_mold_right   = not_special & np.isclose(tri_c[:, 0], W, atol=tol_c)
is_mold_top     = not_special & np.isclose(tri_c[:, 1], H, atol=tol_c)
is_mold_bot     = not_special & np.isclose(tri_c[:, 1], 0, atol=tol_c) & (r_tri > R_paste - tol_r)
is_paste_bot    = not_special & np.isclose(tri_c[:, 1], 0, atol=tol_c) & \
                  (r_tri > R_tube - tol_r) & (r_tri < R_paste + tol_r) & ~is_mold_bot
is_tube_exposed = not_special & np.isclose(r_tri, R_tube, atol=tol_r) & (tri_c[:, 1] < -tol_c)
is_robin = is_z_end | is_mold_right | is_mold_top | is_mold_bot | is_paste_bot | is_tube_exposed

tri_dir = all_tris[is_dirichlet]
tri_rob = all_tris[is_robin]

print(f"\nMesh: {Nn} nodes, {len(all_tets)} tets")
print(f"  tube: {len(tets_tube)}, paste: {len(tets_paste)}, mold: {len(tets_mold)}")
print(f"  Dirichlet: {len(tri_dir)}, Robin: {len(tri_rob)}, Sym: {int(is_sym.sum())}")

# ================================================================
# 5) VECTORIZED FEM ASSEMBLY  (K + M per material)
# ================================================================
f_full = np.zeros(Nn)
K_rows, K_cols, K_vals = [], [], []
M_rows, M_cols, M_vals = [], [], []

G_ref = np.array([[-1., -1., -1.],
                   [ 1.,  0.,  0.],
                   [ 0.,  1.,  0.],
                   [ 0.,  0.,  1.]])

# Consistent mass template for linear tet:
# M_e = rho*cp * V/20 * [[2,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2]]
M_template = np.array([[2,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2]], dtype=float)


def assemble_KM_tet(tets, k_val, rho_val, cp_val):
    if len(tets) == 0:
        return (np.array([]),) * 6
    i1, i2, i3, i4 = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]
    d1 = P[i2] - P[i1]; d2 = P[i3] - P[i1]; d3 = P[i4] - P[i1]
    J = np.stack([d1, d2, d3], axis=2)
    detJ = np.linalg.det(J)
    V = np.abs(detJ) / 6.0
    valid = V > 0
    J_v, V_v, tets_v = J[valid], V[valid], tets[valid]

    J_inv = np.linalg.inv(J_v)
    G_phys = np.einsum('ij,ejk->eik', G_ref, J_inv)

    # Stiffness
    Ke = k_val * V_v[:, None, None] * np.einsum('eij,ekj->eik', G_phys, G_phys)
    # Mass
    Me = (rho_val * cp_val * V_v / 20.0)[:, None, None] * M_template[None, :, :]

    nd = tets_v
    rows = np.repeat(nd, 4, axis=1).ravel()
    cols = np.tile(nd, (1, 4)).ravel()
    return rows, cols, Ke.ravel(), rows.copy(), cols.copy(), Me.ravel()


for tets, k, rho, cp, name in [
    (tets_tube, k_tube, rho_tube, cp_tube, "tube"),
    (tets_paste, k_paste, rho_paste, cp_paste, "paste"),
    (tets_mold, k_solder, rho_solder, cp_solder, "mold"),
]:
    kr, kc, kv, mr, mc, mv = assemble_KM_tet(tets, k, rho, cp)
    K_rows.append(kr); K_cols.append(kc); K_vals.append(kv)
    M_rows.append(mr); M_cols.append(mc); M_vals.append(mv)
    print(f"  assembled {name}: {len(tets)} tets")

# Robin BC
H_tmpl = np.array([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]])
if len(tri_rob) > 0:
    a, b, c = tri_rob[:, 0], tri_rob[:, 1], tri_rob[:, 2]
    cross = np.cross(P[b] - P[a], P[c] - P[a])
    A_tri = 0.5 * np.sqrt(np.sum(cross**2, axis=1))
    coeff = h_conv * A_tri / 12.0
    Kh = coeff[:, None, None] * H_tmpl[None, :, :]
    nd = tri_rob
    rows_r = np.repeat(nd, 3, axis=1).ravel()
    cols_r = np.tile(nd, (1, 3)).ravel()
    K_rows.append(rows_r); K_cols.append(cols_r); K_vals.append(Kh.ravel())
    fh = h_conv * Tinf * A_tri / 3.0
    np.add.at(f_full, a, fh); np.add.at(f_full, b, fh); np.add.at(f_full, c, fh)

# Build sparse K and M
def build_sparse(rows_l, cols_l, vals_l):
    r = np.concatenate(rows_l).astype(int)
    c = np.concatenate(cols_l).astype(int)
    v = np.concatenate(vals_l)
    return coo_matrix((v, (r, c)), shape=(Nn, Nn)).tocsr()

K_full = build_sparse(K_rows, K_cols, K_vals)
M_full = build_sparse(M_rows, M_cols, M_vals)

# ================================================================
# 6) DIRICHLET SUBSPACE + FACTORIZE
# ================================================================
dir_nodes = np.unique(tri_dir.ravel())
free_mask = np.ones(Nn, dtype=bool)
free_mask[dir_nodes] = False
free_idx = np.where(free_mask)[0]
Nf, Nd = len(free_idx), len(dir_nodes)

K_ff = K_full[free_idx, :][:, free_idx]
M_ff = M_full[free_idx, :][:, free_idx]

K_fd_g = K_full[free_idx, :][:, dir_nodes] @ np.full(Nd, T_wall)
f_free_static = f_full[free_idx] - K_fd_g

A_ff = (M_ff / dt) + K_ff
print(f"\nDirichlet: {Nd}, Free: {Nf}")
print("Factorizing system matrix ...")
t0 = timer.time()
solve_A = factorized(A_ff.tocsc())
print(f"Factorized in {timer.time()-t0:.1f}s")

# ================================================================
# 7) TIME STEPPING with convergence auto-stop
# ================================================================
Tn_free = np.full(Nf, T0, dtype=float)
n_max = int(round(t_end / dt))

tmax_list, times_list = [], []
converged = False
conv_step = -1

# Save VTU snapshots at these times
vtu_times = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
vtu_set = set(int(round(t / dt)) for t in vtu_times)
vtu_set.add(0)

print(f"\nTime stepping: up to {n_max} steps ...")
t_start = timer.time()

for n in range(n_max + 1):
    t = n * dt
    Tn_full = np.full(Nn, T_wall, dtype=float)
    Tn_full[free_idx] = Tn_free
    tmax_list.append(float(Tn_full.max()))
    times_list.append(t)

    # Save VTU snapshot
    if n in vtu_set:
        vtu_path = os.path.join(save_dir, f"Transient3D_3mat_t{t:.3f}.vtu")
        # (skip actual VTU write for speed — only write final)

    if n % 200 == 0:
        elapsed = timer.time() - t_start
        print(f"  step {n:>6d}  t={t:8.3f}s  Tmax={tmax_list[-1]:.6f}C  [{elapsed:.1f}s]")

    if n == n_max:
        break

    # Backward Euler step
    rhs = f_free_static + (M_ff / dt) @ Tn_free
    Tn_free_new = solve_A(rhs)

    dT_max = np.max(np.abs(Tn_free_new - Tn_free))
    Tn_free = Tn_free_new

    if dT_max < conv_tol and n > 0:
        conv_step = n + 1
        t_conv = (n + 1) * dt
        Tn_full = np.full(Nn, T_wall, dtype=float)
        Tn_full[free_idx] = Tn_free
        tmax_list.append(float(Tn_full.max()))
        times_list.append(t_conv)
        converged = True
        elapsed = timer.time() - t_start
        print(f"\n  >>> CONVERGED at step {conv_step}, t={t_conv:.3f}s [{elapsed:.1f}s]")
        print(f"      max|dT| = {dT_max:.2e} < tol={conv_tol}")
        break

tmax_hist = np.array(tmax_list)
times_arr = np.array(times_list)

Tn_full = np.full(Nn, T_wall, dtype=float)
Tn_full[free_idx] = Tn_free

print(f"\nFinal: T_min={Tn_full.min():.4f}C, T_max={Tn_full.max():.4f}C")
if converged:
    print(f"Converged at t={times_arr[-1]:.3f}s (step {conv_step}/{n_max})")
else:
    print(f"Reached t_end={t_end}s without convergence")

# ================================================================
# 8) SAVE FINAL VTU (mirrored full domain)
# ================================================================
on_sym = np.isclose(P[:, 0], xc, atol=1e-12)
non_sym = ~on_sym
n_mirror = np.sum(non_sym)

P_full_pts = np.empty((Nn + n_mirror, 3))
P_full_pts[:Nn] = P
P_full_pts[Nn:, 0] = 2.0 * xc - P[non_sym, 0]
P_full_pts[Nn:, 1] = P[non_sym, 1]
P_full_pts[Nn:, 2] = P[non_sym, 2]

T_full_all = np.empty(Nn + n_mirror)
T_full_all[:Nn] = Tn_full
T_full_all[Nn:] = Tn_full[non_sym]

mirror_map = np.arange(Nn, dtype=int)
mirror_map[non_sym] = Nn + np.arange(n_mirror)

tets_m = np.column_stack([
    mirror_map[all_tets[:, 0]], mirror_map[all_tets[:, 2]],
    mirror_map[all_tets[:, 1]], mirror_map[all_tets[:, 3]]])
tets_full = np.vstack([all_tets, tets_m])

# Write VTU
def write_vtu(points, tets, scalars_dict, filename):
    Np, Nc = points.shape[0], tets.shape[0]
    with open(filename, "w", encoding="utf-8") as fout:
        fout.write('<?xml version="1.0"?>\n')
        fout.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        fout.write('  <UnstructuredGrid>\n')
        fout.write(f'    <Piece NumberOfPoints="{Np}" NumberOfCells="{Nc}">\n')
        fout.write('      <PointData>\n')
        for nm, vals in scalars_dict.items():
            fout.write(f'        <DataArray type="Float64" Name="{nm}" format="ascii">\n')
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
        fout.write("          " + " ".join(str(i*4) for i in range(1, Nc+1)) + "\n")
        fout.write('        </DataArray>\n')
        fout.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        fout.write("          " + " ".join("10" for _ in range(Nc)) + "\n")
        fout.write('        </DataArray>\n')
        fout.write('      </Cells>\n')
        fout.write('    </Piece>\n')
        fout.write('  </UnstructuredGrid>\n')
        fout.write('</VTKFile>\n')

vtu_final = os.path.join(save_dir, "Transient3D_Twall_3mat_final.vtu")
write_vtu(P_full_pts, tets_full, {"T": T_full_all, "dT": T_full_all - Tinf}, vtu_final)
print(f"VTU saved: {vtu_final}")

# ================================================================
# 9) CONVERGENCE PLOT (matplotlib)
# ================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times_arr, tmax_hist, "b-", lw=1.2)
ax.axhline(T_wall, color="r", ls="--", alpha=0.6, label=f"T_wall={T_wall}C")
ax.axhline(Tinf,   color="b", ls="--", alpha=0.6, label=f"Tinf={Tinf}C")
if converged:
    ax.axvline(times_arr[-1], color="g", ls="-.", alpha=0.7,
               label=f"Converged t={times_arr[-1]:.2f}s")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Max temperature (C)")
ax.set_title(f"3D Transient: Tmax vs time (T_wall={T_wall}C)")
ax.legend()
ax.grid(True, alpha=0.3)
conv_fig = os.path.join(save_dir, "Twall_3mat_3D_Tmax_vs_time.png")
plt.savefig(conv_fig, dpi=300, bbox_inches="tight")
plt.close()
print(f"Convergence plot: {conv_fig}")

# ================================================================
# 10) PYVISTA VISUALIZATION (if available)
# ================================================================
try:
    import pyvista as pv
    HAVE_PV = True
except ImportError:
    HAVE_PV = False
    print("PyVista not installed. Open VTU in ParaView.")

if HAVE_PV:
    Pmm = P_full_pts * 1e3
    cells = np.hstack([np.full((tets_full.shape[0], 1), 4, dtype=np.int64), tets_full]).ravel()
    celltypes = np.full(tets_full.shape[0], pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, Pmm)
    grid.point_data["T"]  = T_full_all
    grid.point_data["dT"] = T_full_all - Tinf

    def robust_clim(vals, plo=1.0, phi=99.0, min_span=1e-6):
        lo, hi = np.percentile(vals, [plo, phi])
        if hi - lo < min_span:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 0.5*min_span, mid + 0.5*min_span
        pad = 0.02 * (hi - lo)
        return (lo - pad, hi + pad)

    # Temperature preview
    surf_all = grid.extract_surface(algorithm=None).clean()
    T_surf = np.asarray(surf_all.point_data["T"], float)
    T_CLIM = robust_clim(T_surf)

    status = f"CONVERGED t={times_arr[-1]:.2f}s" if converged else f"t={times_arr[-1]:.1f}s"
    p1 = pv.Plotter()
    p1.add_text(f"Transient 3D final state ({status})", font_size=10)
    p1.add_mesh(surf_all, scalars="T", preference="point",
                cmap=cmap_name, clim=T_CLIM, show_edges=False,
                scalar_bar_args={"title": "T (C)", "fmt": "%.3g"})
    p1.view_isometric()
    p1.camera.zoom(1.2)
    p1.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
    p1.show()

    # Interactive slicer
    xmin, xmax = float(Pmm[:,0].min()), float(Pmm[:,0].max())
    ymin, ymax = float(Pmm[:,1].min()), float(Pmm[:,1].max())
    zmin, zmax = float(Pmm[:,2].min()), float(Pmm[:,2].max())
    x0, y0, z0 = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)
    cut = {"x": x0, "y": y0, "z": z0}

    p2 = pv.Plotter()
    p2.add_text("Interactive slicer: Body=T, Slices=dT", font_size=10)
    actors = {"body": None, "s0": None, "s1": None, "s2": None}

    def redraw():
        for key in actors:
            if actors[key] is not None:
                p2.remove_actor(actors[key])
        vol = grid
        vol = vol.clip(normal=(1,0,0), origin=(cut["x"],0,0), invert=False)
        vol = vol.clip(normal=(0,1,0), origin=(0,cut["y"],0), invert=False)
        vol = vol.clip(normal=(0,0,1), origin=(0,0,cut["z"]), invert=False)
        body = vol.extract_surface(algorithm=None).clean()
        slices = [
            grid.slice(normal=(1,0,0), origin=(cut["x"],0,0)),
            grid.slice(normal=(0,1,0), origin=(0,cut["y"],0)),
            grid.slice(normal=(0,0,1), origin=(0,0,cut["z"])),
        ]
        if body is not None and body.n_points > 0:
            tc = robust_clim(np.asarray(body.point_data["T"], float))
            actors["body"] = p2.add_mesh(body, scalars="T", preference="point",
                                          cmap=cmap_name, clim=tc, name="body",
                                          scalar_bar_args={"title":"T(C)","fmt":"%.3g"})
        dT_clim = robust_clim(grid.point_data["dT"])
        for i, sl in enumerate(slices):
            if sl is not None and sl.n_points > 0:
                actors[f"s{i}"] = p2.add_mesh(
                    sl, scalars="dT", preference="point", cmap=cmap_name,
                    clim=dT_clim, name=f"sl{i}", reset_camera=False,
                    show_scalar_bar=(i==0),
                    scalar_bar_args={"title":"dT(C)","fmt":"%.3g"})

    redraw()
    def sx(v): cut["x"]=float(v); redraw()
    def sy(v): cut["y"]=float(v); redraw()
    def sz(v): cut["z"]=float(v); redraw()
    p2.add_slider_widget(sx, [xmin,xmax], value=x0, title="X(mm)", pointa=(.03,.10), pointb=(.35,.10))
    p2.add_slider_widget(sy, [ymin,ymax], value=y0, title="Y(mm)", pointa=(.03,.06), pointb=(.35,.06))
    p2.add_slider_widget(sz, [zmin,zmax], value=z0, title="Z(mm)", pointa=(.03,.02), pointb=(.35,.02))
    p2.view_isometric(); p2.camera.zoom(1.2)
    p2.show_grid(xtitle="X(mm)", ytitle="Y(mm)", ztitle="Z(mm)")
    p2.show()

print("\nDone.")
