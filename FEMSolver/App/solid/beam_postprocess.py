"""
Post-processing: plots and analytical comparisons for beam analysis.

Analytical spring constants
----------------------------
Straight cantilever
  I   = t · W³ / 12          (bending about z, load in y)
  k   = 3 E I / L³

Zig-zag (n_segs segments in series, each length L_seg, width W):
  "Free-end"  model (cantilever BCs per segment):
      k_free   = 3 E I / (n_segs · L_seg³)

  "Guided-end" model (zero-slope BCs at both ends):
      k_guided = 12 E I / (n_segs · L_seg³)

The FEM result should fall between k_free and k_guided (closer to
k_guided when the connectors are stiff relative to the segments).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec


# ── dark-theme axis helper ────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor("#1e1e1e")
    ax.tick_params(colors="#ccc")
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    ax.title.set_color("#ddd")
    for sp in ax.spines.values():
        sp.set_color("#555")


# ── analytical formulae ───────────────────────────────────────────────────────

def analytical_straight(L, W, t, E):
    """Spring constant for straight cantilever (load in y, width=W, thickness=t)."""
    I = t * W**3 / 12.0
    k = 3.0 * E * I / L**3
    delta_formula = f"k = 3EI/L³ = 3·E·t·W³/(12·L³)"
    return k, I, delta_formula


def analytical_zigzag(L_seg, W, t, E, n_segs):
    """Free-end and guided-end spring constants for zig-zag beam."""
    I = t * W**3 / 12.0
    k_free   = 3.0  * E * I / (n_segs * L_seg**3)
    k_guided = 12.0 * E * I / (n_segs * L_seg**3)
    return k_free, k_guided, I


# ── displacement magnitude (node-wise) ───────────────────────────────────────

def displacement_magnitude(u):
    """Euclidean displacement at each node, shape (Nn,)."""
    ux = u[0::2]
    uy = u[1::2]
    return np.sqrt(ux**2 + uy**2)


# ── main result figure ────────────────────────────────────────────────────────

def plot_beam_results(coords, tris, u, svm,
                      beam_type, params, results,
                      scale_factor=None, fig=None):
    """
    Two-panel result figure:
      Left  — colour map of |u| on deformed mesh
      Right — von-Mises stress on undeformed mesh

    Parameters
    ----------
    coords    : (Nn,2) undeformed coordinates
    tris      : (Ne,3) connectivity
    u         : (2Nn,) displacement vector
    svm       : (Ne,)  von-Mises stress per element
    beam_type : "straight" or "zigzag"
    params    : dict with keys L, W, t, E, nu, F, (g, n_segs for zigzag)
    results   : dict with keys k_fem, delta, k_theory (str or float), error_pct
    scale_factor : visual displacement amplification (None → auto)

    Returns
    -------
    fig : matplotlib Figure
    """
    ux = u[0::2]
    uy = u[1::2]
    mag = np.sqrt(ux**2 + uy**2)

    if scale_factor is None:
        char_len = max(params.get("L", 1.0), params.get("W", 1.0))
        max_disp = mag.max()
        if max_disp > 1e-30:
            scale_factor = 0.1 * char_len / max_disp
        else:
            scale_factor = 1.0

    coords_def = coords.copy()
    coords_def[:, 0] += scale_factor * ux
    coords_def[:, 1] += scale_factor * uy

    if fig is None:
        fig = plt.figure(figsize=(14, 6), facecolor="#2a2a2a")

    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    _style(ax1);  _style(ax2)

    # ── left: displacement magnitude on deformed mesh ──────────────────────
    tri_def = mtri.Triangulation(coords_def[:, 0], coords_def[:, 1], tris)
    cf1 = ax1.tricontourf(tri_def, mag * 1e6, levels=40, cmap="plasma")
    ax1.triplot(tri_def, linewidth=0.15, alpha=0.15, color="white")
    cb1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
    cb1.set_label("|u| (µm)", color="#ccc")
    cb1.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color="#ccc")

    scale_txt = f"(×{scale_factor:.0f} deformation)"
    _W = params.get("W", 0)
    _L = params.get("L", 0)
    ax1.set_title(f"Displacement magnitude {scale_txt}", color="#ddd")
    ax1.set_xlabel("x (m)");  ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.15)

    # ── right: von-Mises stress on original mesh ───────────────────────────
    # Map element-wise SVM → node-wise (simple averaging)
    Nn = coords.shape[0]
    node_svm   = np.zeros(Nn)
    node_count = np.zeros(Nn)
    for e, (n0, n1, n2) in enumerate(tris):
        node_svm[n0]   += svm[e]; node_count[n0] += 1
        node_svm[n1]   += svm[e]; node_count[n1] += 1
        node_svm[n2]   += svm[e]; node_count[n2] += 1
    node_count = np.maximum(node_count, 1)
    node_svm /= node_count

    tri_orig = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)
    cf2 = ax2.tricontourf(tri_orig, node_svm / 1e6, levels=40, cmap="hot")
    ax2.triplot(tri_orig, linewidth=0.15, alpha=0.15, color="white")
    cb2 = fig.colorbar(cf2, ax=ax2, pad=0.03)
    cb2.set_label("von-Mises (MPa)", color="#ccc")
    cb2.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="#ccc")

    ax2.set_title("von-Mises Stress", color="#ddd")
    ax2.set_xlabel("x (m)");  ax2.set_ylabel("y (m)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.15)

    # ── figure subtitle with key numbers ──────────────────────────────────
    k_fem   = results.get("k_fem", 0.0)
    delta   = results.get("delta", 0.0)
    err     = results.get("error_pct", None)
    k_theo  = results.get("k_theory", None)

    lines = [
        f"d_tip = {delta*1e6:.4f} um     k_FEM = {k_fem:.4f} N/m",
    ]
    if k_theo is not None:
        if isinstance(k_theo, (int, float)):
            lines.append(f"k_theory = {k_theo:.4f} N/m     "
                         f"error = {err:.2f}%")
        else:
            lines.append(str(k_theo))

    fig.suptitle("\n".join(lines), color="#aaa", fontsize=10,
                 fontfamily="monospace", y=0.02)

    return fig


# ── text summary ──────────────────────────────────────────────────────────────

def result_summary(beam_type, params, results):
    """Return a multi-line plain-text summary string."""
    sep = "-" * 55
    lines = [
        sep,
        f"  Beam type : {beam_type.upper()}",
        sep,
    ]

    L = params.get("L", 0); W = params.get("W", 0)
    t = params.get("t", 0); E = params.get("E", 0)
    nu = params.get("nu", 0); F = params.get("F", 0)

    lines += [
        f"  L  = {L*1e6:.2f} um     W = {W*1e6:.2f} um     t = {t*1e6:.2f} um",
        f"  E  = {E/1e9:.2f} GPa    nu = {nu}",
        f"  F  = {F:.3e} N",
    ]

    if beam_type == "zigzag":
        g      = params.get("g", 0)
        n_segs = params.get("n_segs", 1)
        lines += [
            f"  g  = {g*1e6:.2f} um    n_segs = {n_segs}",
        ]

    lines.append(sep)

    k_fem  = results.get("k_fem", 0)
    delta  = results.get("delta", 0)
    lines += [
        f"  d_tip (FEM)  = {delta*1e6:.6f} um",
        f"  k_FEM        = {k_fem:.6f} N/m",
    ]

    if beam_type == "straight":
        k_th  = results.get("k_theory", None)
        err   = results.get("error_pct", None)
        if k_th is not None:
            lines += [
                f"  k_theory     = {k_th:.6f} N/m   (3EI/L^3)",
                f"  Error        = {err:.3f}%",
            ]
    else:
        k_free   = results.get("k_free",   None)
        k_guided = results.get("k_guided", None)
        if k_free is not None:
            lines += [
                f"  k_free-end   = {k_free:.6f} N/m  (lower bound)",
                f"  k_guided-end = {k_guided:.6f} N/m  (upper bound)",
                f"  FEM within bounds: "
                f"{'YES' if k_free <= k_fem <= k_guided else 'NO'}",
            ]

    lines.append(sep)
    return "\n".join(lines)
