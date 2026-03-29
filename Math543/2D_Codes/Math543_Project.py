# ================================================================
# Math 543 – Boundary Value Problems Project
# FIGURE 1: Problem statement (NO SOLUTION)
#
# Geometry (half-domain by symmetry):
#   Rectangle: 0 ≤ y ≤ H,  xc ≤ x ≤ W,  where xc = W/2
#   Quarter-circle cut: (x-xc)^2 + y^2 = R^2   (cut-out boundary is the arc)
#
# Boundary conditions shown:
#   • Heat flux INTO solid on arc (Neumann):      -k ∂T/∂n = q''_in
#   • Convection on outer boundaries (Robin):     -k ∂T/∂n = h (T - T_inf)
#   • Symmetry line insulated:                    ∂T/∂n = 0
#
# Output:
#   A readable schematic figure with arrows + labels, saved as:
#       Problem_statement.png
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------- PARAMETERS (EDIT HERE) ---------------------
# Geometry (mm → meters)
W_mm = 1.2          # full width (mm)
H_mm = 1.0          # height (mm)
R_mm = 0.5          # semicircle radius (mm) -> quarter-circle in half-domain

# Ambient / convection (for labeling only)
T_inf = 25.0        # °C
h = 15.0            # W/m^2-K

# Heat input on FULL semicircle, converted to flux q'' (W/m^2)
use_total_power = True
P_total = 50.0      # total power entering FULL semicircle (W)
b_mm = 1.0          # thickness into page (mm)

# Direct heat flux option (used if use_total_power = False)
q_in_direct = 1e5   # W/m^2

# Arrow visualization
arrow_len = 0.08e-3
n_edge = 4
n_arc = 6
# ---------------------------------------------------------------

# Unit conversions
W = W_mm * 1e-3
H = H_mm * 1e-3
R = R_mm * 1e-3
xc = (W_mm / 2) * 1e-3
b = b_mm * 1e-3

# Convert power → heat flux (uniform on arc)
# Full semicircle arc length = pi*R, area = (pi*R)*b
if use_total_power:
    q_in = P_total / (np.pi * R * b)
else:
    q_in = q_in_direct

# -------------------- FIGURE SETUP ---------------------
fig, ax = plt.subplots(figsize=(6, 10))  # tall to avoid compression

# -------------------- GEOMETRY -------------------------
# Rectangle (half-domain)
rect_x = [xc, W, W, xc, xc]
rect_y = [0, 0, H, H, 0]
ax.plot(rect_x, rect_y, 'k', linewidth=2)

# Quarter-circle cut boundary (arc)
theta = np.linspace(0, np.pi/2, 300)
x_arc = xc + R * np.cos(theta)
y_arc = R * np.sin(theta)
ax.plot(x_arc, y_arc, 'k', linewidth=2)

# -------------------- CONVECTION ARROWS (heat loss OUT) ----------
# Right wall (outward normal +x)
ys = np.linspace(0.2 * H, 0.8 * H, n_edge)
ax.quiver(W * np.ones_like(ys), ys,
          arrow_len, 0,
          angles='xy', scale_units='xy', scale=1)

# Top wall (outward normal +y)
xs = np.linspace(xc + 0.15 * (W - xc), W - 0.15 * (W - xc), n_edge)
ax.quiver(xs, H * np.ones_like(xs),
          0, arrow_len,
          angles='xy', scale_units='xy', scale=1)

# Bottom remaining segment (outward normal -y), only x in [xc+R, W]
xs = np.linspace(xc + R + 0.10 * (W - (xc + R)),
                 W - 0.10 * (W - (xc + R)),
                 n_edge)
ax.quiver(xs, np.zeros_like(xs),
          0, -arrow_len,
          angles='xy', scale_units='xy', scale=1)

# -------------------- HEAT FLUX ARROWS (IN on arc) ---------------
theta_a = np.linspace(0.15, 0.85 * (np.pi / 2), n_arc)
x_a = xc + R * np.cos(theta_a)
y_a = R * np.sin(theta_a)

# Use radial direction to show "into solid" arrows from the cut boundary
nx = np.cos(theta_a)
ny = np.sin(theta_a)

ax.quiver(x_a, y_a,
          arrow_len * nx, arrow_len * ny,
          angles='xy', scale_units='xy', scale=1)

# -------------------- SYMMETRY LINE ------------------------------
ax.plot([xc, xc], [0, H], 'k--', linewidth=1.5)
ax.text(xc + 0.00002, 0.00105,
        'Symmetry\n$\\partial T/\\partial n = 0$',
        fontsize=12)

# -------------------- LABELS (placed to avoid clipping) ----------
# Convection label near upper right, but inside visible y-range
ax.text(W + 0.00012, H - 0.00025,
        r'Convection (outer boundaries)' '\n'
        r'$-k\,\partial T/\partial n = h\,(T-T_\infty)$' '\n'
        f'$T_\\infty = {T_inf:.0f}^\\circ$C,\  $h = {h:.0f}$',
        fontsize=12, va='top')

# Flux label (inside)
ax.text(xc + 0.00005, 0.0006,
        r'Heat flux IN on arc' '\n'
        r'$-k\,\partial T/\partial n = q''''_{in}$' '\n'
        f'$q''''_{{in}} = {q_in:.2e}$ W/m$^2$',
        fontsize=12)

# -------------------- FINAL STYLING ------------------------------
ax.set_aspect('equal')

# Add margin so top/bottom text doesn't clip
ax.set_xlim(xc - 0.00005, W + 0.00035)
ax.set_ylim(-0.00020, H + 0.00020)

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Problem statement', fontsize=16)
ax.grid(True)

# Save high-quality figure
plt.savefig('Problem_statement.png', dpi=300, bbox_inches='tight')
plt.show()
