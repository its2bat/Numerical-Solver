# ================================================================
# FEM Heat Transfer – Boundary Value Problems Project
# FIGURE 2: Mesh (NO SOLUTION)
#
# Half-domain geometry:
#   x in [xc, W], y in [0, H],  where xc = W/2
#   Quarter-circle cut boundary: (x-xc)^2 + y^2 = R^2
#
# Output:
#   Mesh.png
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------- PARAMETERS (EDIT HERE) ---------------------
# Geometry (mm → meters)
W_mm = 1.2
H_mm = 2.0
R_mm = 0.5

# Mesh control
h_global = 0.06e-3     # typical element size (m) ~ 0.06 mm
h_arc    = 0.02e-3     # finer size near the arc (m) ~ 0.02 mm (optional)

# Plot options
show_node_points = False   # set True if you want to see nodes
# ---------------------------------------------------------------

W  = W_mm * 1e-3
H  = H_mm * 1e-3
R  = R_mm * 1e-3
xc = (W_mm / 2) * 1e-3

# --- Build mesh with gmsh ---
import gmsh

gmsh.initialize()
gmsh.model.add("half_domain_quarter_cut")

# Points on boundary (ordered around domain)
# p1 = (xc+R, 0)    intersection with bottom
# p2 = (W, 0)
# p3 = (W, H)
# p4 = (xc, H)
# p5 = (xc, R)      intersection with symmetry line
# center = (xc, 0)  for quarter-circle arc
p1 = gmsh.model.geo.addPoint(xc + R, 0.0, 0.0, h_arc)
p2 = gmsh.model.geo.addPoint(W,      0.0, 0.0, h_global)
p3 = gmsh.model.geo.addPoint(W,      H,   0.0, h_global)
p4 = gmsh.model.geo.addPoint(xc,     H,   0.0, h_global)
p5 = gmsh.model.geo.addPoint(xc,     R,   0.0, h_arc)
pc = gmsh.model.geo.addPoint(xc,     0.0, 0.0, h_arc)

# Boundary curves
l1 = gmsh.model.geo.addLine(p1, p2)           # bottom (remaining segment)
l2 = gmsh.model.geo.addLine(p2, p3)           # right wall
l3 = gmsh.model.geo.addLine(p3, p4)           # top wall
l4 = gmsh.model.geo.addLine(p4, p5)           # symmetry wall (down to arc start)
arc = gmsh.model.geo.addCircleArc(p5, pc, p1) # quarter-circle cut arc

# Surface
cloop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, arc])
surf  = gmsh.model.geo.addPlaneSurface([cloop])

gmsh.model.geo.synchronize()

# (Optional) extra refinement near arc using a distance field
# This makes the mesh smoother/refined near the arc even if global size is larger.
field_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(field_dist, "CurvesList", [arc])
gmsh.model.mesh.field.setNumber(field_dist, "Sampling", 100)

field_thresh = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field_thresh, "InField", field_dist)
gmsh.model.mesh.field.setNumber(field_thresh, "SizeMin", h_arc)
gmsh.model.mesh.field.setNumber(field_thresh, "SizeMax", h_global)
gmsh.model.mesh.field.setNumber(field_thresh, "DistMin", 0.05e-3)
gmsh.model.mesh.field.setNumber(field_thresh, "DistMax", 0.20e-3)

gmsh.model.mesh.field.setAsBackgroundMesh(field_thresh)

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# --- Extract nodes and triangles ---
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
coords = node_coords.reshape(-1, 3)[:, :2]  # (N,2)

# Get triangle elements on our surface
elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surf)

# Find 3-node triangle type (=2 in gmsh)
tri_conn = None
for etype, enodes in zip(elem_types, elem_node_tags):
    if etype == 2:  # 3-node triangle
        tri_conn = enodes.reshape(-1, 3)
        break

if tri_conn is None:
    gmsh.finalize()
    raise RuntimeError("No 3-node triangles found (etype=2). Try different extraction logic.")

# Map gmsh node tags → 0-based indices for plotting
tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
triangles = np.vectorize(tag_to_idx.get)(tri_conn)

gmsh.finalize()

# --- Plot mesh ---
fig, ax = plt.subplots(figsize=(6, 10))

# Draw triangle edges
for tri in triangles:
    pts = coords[tri]
    poly = np.vstack([pts, pts[0]])
    ax.plot(poly[:, 0], poly[:, 1], linewidth=0.4)

if show_node_points:
    ax.scatter(coords[:, 0], coords[:, 1], s=2)

ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Mesh")
ax.grid(True, alpha=0.3)

# Add comfortable margins
ax.set_xlim(xc - 0.00005, W + 0.00035)
ax.set_ylim(-0.00020, H + 0.00020)

plt.savefig("Mesh.png", dpi=300, bbox_inches="tight")
plt.show()
