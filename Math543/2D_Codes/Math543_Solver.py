# ================================================================
# Math 543 – BVP Project
# STEADY-STATE 2D HEAT CONDUCTION (FEM) on Gmsh mesh (half-domain)
# + Full-domain visualization by symmetry mirroring
#
# Domain (half by symmetry):
#   Rectangle:  xc <= x <= W , 0 <= y <= H
#   Quarter-circle cut: (x-xc)^2 + y^2 = R^2   (arc boundary)
#
# Governing equation (steady, no internal generation):
#   div(k grad T) = 0   in Ω
#
# Boundary conditions:
#   ARC   (Neumann flux IN):   -k dT/dn = q_in
#   OUTER (Robin convection):  -k dT/dn = h (T - Tinf)
#   SYM   (Symmetry):          dT/dn = 0
#
# Outputs:
#   Temperature_half.png   (half-domain)
#   Temperature_full.png   (full-domain mirrored)
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
cut_diameter_mm = 1.0
R_mm = cut_diameter_mm / 2.0  # quarter-circle radius in HALF domain

# Material (stainless-ish)
k = 16.0  # W/m-K

# Convection
h = 15.0     # W/m^2-K
Tinf = 25.0  # °C

# Heat flux on ARC (direct)
q_in = 1.0e3  # W/m^2 (1 kW/m^2)

# Mesh sizing
h_global = 0.06e-3
h_arc    = 0.02e-3
dist_min = 0.04e-3
dist_max = 0.18e-3

# Output names
half_fig_name = "Temperature_half.png"
full_fig_name = "Temperature_full.png"
# ---------------------------------------------------------------

# Unit conversions
W  = W_mm * 1e-3
H  = H_mm * 1e-3
R  = R_mm * 1e-3
xc = (W_mm / 2) * 1e-3

# Geometry validity checks for half-domain quarter-circle cut
if R > (W - xc) + 1e-15:
    raise ValueError(
        f"Invalid geometry: need R <= W-xc = W/2. "
        f"Got R={R_mm:.3f} mm, W/2={(W_mm/2):.3f} mm. "
        f"Increase W or reduce R."
    )
if R > H + 1e-15:
    raise ValueError(
        f"Invalid geometry: need R <= H. Got R={R_mm:.3f} mm, H={H_mm:.3f} mm."
    )

print(f"Using q_in = {q_in:.3e} W/m^2 (direct)")
print(f"Geometry: W={W_mm} mm, H={H_mm} mm, cut diameter={cut_diameter_mm} mm (R={R_mm} mm)")

# -------------------- BUILD MESH WITH PHYSICAL TAGS --------------------
# Spyder re-run safety
try:
    gmsh.finalize()
except Exception:
    pass

gmsh.initialize()
gmsh.model.add("steady_heat_half_domain")

# Points
p1 = gmsh.model.geo.addPoint(xc + R, 0.0, 0.0, h_arc)     # bottom-arc intersection
p2 = gmsh.model.geo.addPoint(W,      0.0, 0.0, h_global)  # bottom-right
p3 = gmsh.model.geo.addPoint(W,      H,   0.0, h_global)  # top-right
p4 = gmsh.model.geo.addPoint(xc,     H,   0.0, h_global)  # top-sym
p5 = gmsh.model.geo.addPoint(xc,     R,   0.0, h_arc)     # arc-sym intersection
pc = gmsh.model.geo.addPoint(xc,     0.0, 0.0, h_arc)     # circle center

# Curves
l_bottom = gmsh.model.geo.addLine(p1, p2)          # OUTER (bottom remaining)
l_right  = gmsh.model.geo.addLine(p2, p3)          # OUTER
l_top    = gmsh.model.geo.addLine(p3, p4)          # OUTER
l_sym    = gmsh.model.geo.addLine(p4, p5)          # SYM
c_arc    = gmsh.model.geo.addCircleArc(p5, pc, p1) # ARC (flux)

# Surface
cloop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_sym, c_arc])
surf  = gmsh.model.geo.addPlaneSurface([cloop])

gmsh.model.geo.synchronize()

# Physical groups
pg_arc = gmsh.model.addPhysicalGroup(1, [c_arc])
gmsh.model.setPhysicalName(1, pg_arc, "ARC")

pg_outer = gmsh.model.addPhysicalGroup(1, [l_bottom, l_right, l_top])
gmsh.model.setPhysicalName(1, pg_outer, "OUTER")

pg_sym = gmsh.model.addPhysicalGroup(1, [l_sym])
gmsh.model.setPhysicalName(1, pg_sym, "SYM")

pg_dom = gmsh.model.addPhysicalGroup(2, [surf])
gmsh.model.setPhysicalName(2, pg_dom, "DOMAIN")

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

# Generate mesh
gmsh.model.mesh.generate(2)

# -------------------- EXTRACT NODES --------------------
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
coords_all = node_coords.reshape(-1, 3)[:, :2]
tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

# -------------------- TRIANGLES (DOMAIN) --------------------
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

# -------------------- EDGES FROM PHYSICAL GROUPS --------------------
def edges_from_physical_group(phys_id):
    edges = []
    entities = gmsh.model.getEntitiesForPhysicalGroup(1, phys_id)
    for ent in entities:
        etypes, _, enodes = gmsh.model.mesh.getElements(1, ent)
        for etype, nodes in zip(etypes, enodes):
            if etype == 1:  # 2-node line
                conn = nodes.reshape(-1, 2)
                e = np.vectorize(tag_to_idx.get)(conn)
                edges.append(e)
            elif etype == 8:  # 3-node quadratic line -> endpoints only
                conn = nodes.reshape(-1, 3)[:, :2]
                e = np.vectorize(tag_to_idx.get)(conn)
                edges.append(e)

    if not edges:
        return np.empty((0, 2), dtype=int)

    E = np.vstack(edges)
    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)
    return E

edges_arc_all   = edges_from_physical_group(pg_arc)
edges_outer_all = edges_from_physical_group(pg_outer)
edges_sym_all   = edges_from_physical_group(pg_sym)

gmsh.finalize()

print(f"Total nodes (gmsh): {coords_all.shape[0]}")
print(f"Triangles: {len(triangles_all)}")
print(f"ARC edges: {len(edges_arc_all)} | OUTER edges: {len(edges_outer_all)} | SYM edges: {len(edges_sym_all)}")

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
edges_sym = new_index[edges_sym_all]  # natural BC, not used in assembly

Nn = coords.shape[0]
print(f"Active nodes used in FEM: {Nn}")

# -------------------- FEM ASSEMBLY --------------------
I, J, V = [], [], []
f = np.zeros(Nn)

# Element stiffness (P1 triangles)
for tri in triangles:
    i1, i2, i3 = tri
    x1, y1 = coords[i1]
    x2, y2 = coords[i2]
    x3, y3 = coords[i3]

    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * abs(det)
    if A <= 0:
        continue

    bvec = np.array([y2 - y3, y3 - y1, y1 - y2])
    cvec = np.array([x3 - x2, x1 - x3, x2 - x1])
    Ke = (k / (4.0 * A)) * (np.outer(bvec, bvec) + np.outer(cvec, cvec))

    nodes = [i1, i2, i3]
    for a in range(3):
        for b_ in range(3):
            I.append(nodes[a]); J.append(nodes[b_]); V.append(Ke[a, b_])

# Robin convection on OUTER edges
for (i, j) in edges_outer:
    xi, yi = coords[i]
    xj, yj = coords[j]
    L = np.hypot(xj - xi, yj - yi)

    Kh = (h * L / 6.0) * np.array([[2.0, 1.0],
                                  [1.0, 2.0]])
    fh = (h * Tinf * L / 2.0) * np.array([1.0, 1.0])

    f[i] += fh[0]; f[j] += fh[1]
    I += [i, i, j, j]
    J += [i, j, i, j]
    V += [Kh[0, 0], Kh[0, 1], Kh[1, 0], Kh[1, 1]]

# Neumann flux on ARC edges
for (i, j) in edges_arc:
    xi, yi = coords[i]
    xj, yj = coords[j]
    L = np.hypot(xj - xi, yj - yi)

    fq = (q_in * L / 2.0) * np.array([1.0, 1.0])
    f[i] += fq[0]; f[j] += fq[1]

K = coo_matrix((V, (I, J)), shape=(Nn, Nn)).tocsr()

# -------------------- SOLVE (NO ANCHOR) --------------------
T = spsolve(K, f)

print(f"T min = {T.min():.4f} °C, T max = {T.max():.4f} °C")

# -------------------- ENERGY CHECK (per unit thickness) --------------------
qin_line = 0.0
for (i, j) in edges_arc:
    L = np.hypot(*(coords[j] - coords[i]))
    qin_line += q_in * L

qloss_line = 0.0
for (i, j) in edges_outer:
    L = np.hypot(*(coords[j] - coords[i]))
    Tm = 0.5 * (T[i] + T[j])
    qloss_line += h * (Tm - Tinf) * L

print(f"Line heat IN  (per thickness):  {qin_line:.6e} W/m")
print(f"Line heat OUT (per thickness): {qloss_line:.6e} W/m")
if abs(qin_line) > 1e-14:
    print(f"Relative balance error: {(qloss_line - qin_line) / qin_line:.2%}")

# -------------------- COLORMAP (match your convection figure) -----------------
# Turbo is very close to your example: deep blue -> cyan -> green -> yellow -> red
cmap_convection = "turbo"

# -------------------- PLOT HALF-DOMAIN --------------------
triang_half = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

fig, ax = plt.subplots(figsize=(6.2, 5.6))
cf = ax.tricontourf(triang_half, T, levels=40, cmap=cmap_convection)

cbar = fig.colorbar(cf, ax=ax, pad=0.03)
cbar.set_label("Temperature (°C)")

ax.triplot(triang_half, linewidth=0.12, alpha=0.18)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Temperature map (steady state) — half domain")
ax.grid(True, alpha=0.2)

padx = 0.00015
pady = 0.00010
ax.set_xlim(xc - 0.00005, W + padx)
ax.set_ylim(-pady, H + pady)

plt.savefig(half_fig_name, dpi=300, bbox_inches="tight")
plt.show()

# -------------------- FULL-DOMAIN VISUALIZATION (mirror about x=xc) ------------
tol = 1e-12
x = coords[:, 0]
on_sym = np.isclose(x, xc, atol=tol)

N_half = coords.shape[0]
mirror_index = -np.ones(N_half, dtype=int)

coords_full = coords.copy()
T_full = T.copy()

# Add mirrored nodes (excluding symmetry nodes to avoid duplicates)
for i in range(N_half):
    if on_sym[i]:
        mirror_index[i] = i
    else:
        xm = 2.0 * xc - coords[i, 0]
        ym = coords[i, 1]
        mirror_index[i] = coords_full.shape[0]
        coords_full = np.vstack([coords_full, [xm, ym]])
        T_full = np.append(T_full, T[i])

# Triangles: original + mirrored (reverse order for orientation)
tri_m = np.column_stack([
    mirror_index[triangles[:, 0]],
    mirror_index[triangles[:, 2]],
    mirror_index[triangles[:, 1]],
])
triangles_full = np.vstack([triangles, tri_m])

triang_full = mtri.Triangulation(coords_full[:, 0], coords_full[:, 1], triangles_full)

fig, ax = plt.subplots(figsize=(6.8, 5.6))
cf = ax.tricontourf(triang_full, T_full, levels=40, cmap=cmap_convection)

cbar = fig.colorbar(cf, ax=ax, pad=0.03)
cbar.set_label("Temperature (°C)")

ax.triplot(triang_full, linewidth=0.12, alpha=0.18)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Temperature map (steady state) — full domain (symmetry)")
ax.grid(True, alpha=0.2)

ax.set_xlim(-padx, W + padx)
ax.set_ylim(-pady, H + pady)

plt.savefig(full_fig_name, dpi=300, bbox_inches="tight")
plt.show()
